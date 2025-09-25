import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import json
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a wrapper class for SentenceTransformer to make it compatible with LangChain
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts):
        """Embed a list of documents using SentenceTransformer"""
        return self.model.encode(texts, convert_to_numpy=True).tolist()
        
    def embed_query(self, text):
        """Embed a query using SentenceTransformer"""
        return self.model.encode(text, convert_to_numpy=True).tolist()

class AQIVectorDatabase:
    def __init__(self, data_folder, metadata_file, output_folder="vector_db"):
        """
        Initialize the AQI Vector Database creator
        
        Args:
            data_folder: Path to folder containing CSV files for each station
            metadata_file: Path to CSV file containing station metadata
            output_folder: Folder to save the vector database
        """
        self.data_folder = 'data/daily_data'
        self.metadata_file = 'data/metadata.csv'
        self.output_folder = 'vector_db'
        self.embedding_model = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
        self.scaler = MinMaxScaler()
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
    def load_metadata(self):
        """Load and process the metadata file"""
        logger.info("Loading station metadata...")
        self.metadata = pd.read_csv(self.metadata_file)
        # Create a dictionary for quick lookup
        self.station_info = {}
        for _, row in self.metadata.iterrows():
            self.station_info[row['id']] = {
                'name': row['name'],
                'city': row['city'],
                'country': row['country'],
                'latitude': row['latitude'],
                'longitude': row['longitude']
            }
        logger.info(f"Loaded metadata for {len(self.station_info)} stations")
    
    def load_and_process_data(self):
        """Load and process all station data files"""
        logger.info("Loading station data files...")
        all_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_folder}")
        
        logger.info(f"Found {len(all_files)} station files")
        
        # Process files in batches to avoid memory issues
        all_documents = []
        batch_size = 10
        
        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i:i+batch_size]
            batch_documents = []
            
            for file_path in batch_files:
                station_documents = self.process_station_file(file_path)
                batch_documents.extend(station_documents)
            
            all_documents.extend(batch_documents)
            logger.info(f"Processed batch {i//batch_size + 1}/{len(all_files)//batch_size + 1}, total documents: {len(all_documents)}")
        
        return all_documents
    
    def process_station_file(self, file_path):
        """
        Process a single station file and convert it to documents
        
        Returns:
            List of Document objects for the vector store
        """
        try:
            station_name = os.path.basename(file_path).replace('.csv', '')
            df = pd.read_csv(file_path)
            
            # Clean the data
            df = df.dropna(subset=['date', 'aqi_cal'])
            
            if df.empty:
                logger.warning(f"No valid data in {file_path}")
                return []
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Create documents for different time periods
            documents = []
            
            # Get station metadata
            station_id = df['station_id'].iloc[0] if 'station_id' in df.columns else None
            station_metadata = self.station_info.get(station_id, {})
            
            # Documents by month
            monthly_data = df.groupby(df['date'].dt.to_period('M'))
            for period, month_df in monthly_data:
                month_year = period.strftime('%B %Y')
                
                # Calculate monthly statistics
                monthly_stats = {
                    'avg_aqi': month_df['aqi_cal'].mean(),
                    'max_aqi': month_df['aqi_cal'].max(),
                    'min_aqi': month_df['aqi_cal'].min(),
                    'days_above_200': (month_df['aqi_cal'] > 200).sum(),
                    'days_above_300': (month_df['aqi_cal'] > 300).sum()
                }
                
                # Calculate pollutant averages if they exist
                pollutants = ['co', 'no2', 'no', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
                for pollutant in pollutants:
                    if pollutant in month_df.columns:
                        monthly_stats[f'avg_{pollutant}'] = month_df[pollutant].mean()
                
                # Create a textual summary for embedding
                text_summary = f"Air quality data for {station_metadata.get('name', station_name)} in {station_metadata.get('city', 'Unknown City')} for {month_year}. "
                text_summary += f"Average AQI: {monthly_stats['avg_aqi']:.2f}, Maximum AQI: {monthly_stats['max_aqi']:.2f}, Minimum AQI: {monthly_stats['min_aqi']:.2f}. "
                text_summary += f"Days with AQI > 200: {monthly_stats['days_above_200']}, Days with AQI > 300: {monthly_stats['days_above_300']}. "
                
                # Add pollutant information
                for pollutant in pollutants:
                    if f'avg_{pollutant}' in monthly_stats:
                        text_summary += f"Average {pollutant.upper()}: {monthly_stats[f'avg_{pollutant}']:.2f}. "
                
                # Add location info
                if 'latitude' in station_metadata and 'longitude' in station_metadata:
                    text_summary += f"Located at coordinates: {station_metadata['latitude']}, {station_metadata['longitude']}. "
                
                # Create metadata
                metadata = {
                    'station_id': station_id,
                    'station_name': station_metadata.get('name', station_name),
                    'city': station_metadata.get('city', 'Unknown'),
                    'country': station_metadata.get('country', 'IN'),
                    'latitude': station_metadata.get('latitude'),
                    'longitude': station_metadata.get('longitude'),
                    'time_period': month_year,
                    'period_type': 'monthly',
                    'start_date': month_df['date'].min().strftime('%Y-%m-%d'),
                    'end_date': month_df['date'].max().strftime('%Y-%m-%d'),
                    'data_points': len(month_df),
                    **monthly_stats
                }
                
                # Create document
                doc = Document(page_content=text_summary, metadata=metadata)
                documents.append(doc)
            
            # Create seasonal documents (quarterly)
            df['season'] = df['date'].dt.month.apply(lambda x: 
                'Winter' if x in [12, 1, 2] else
                'Spring' if x in [3, 4, 5] else
                'Summer' if x in [6, 7, 8] else
                'Fall'
            )
            df['year'] = df['date'].dt.year
            
            seasonal_data = df.groupby([df['year'], df['season']])
            for (year, season), season_df in seasonal_data:
                
                # Calculate seasonal statistics
                seasonal_stats = {
                    'avg_aqi': season_df['aqi_cal'].mean(),
                    'max_aqi': season_df['aqi_cal'].max(),
                    'min_aqi': season_df['aqi_cal'].min(),
                    'days_above_200': (season_df['aqi_cal'] > 200).sum(),
                    'days_above_300': (season_df['aqi_cal'] > 300).sum()
                }
                
                # Calculate pollutant averages
                for pollutant in pollutants:
                    if pollutant in season_df.columns:
                        seasonal_stats[f'avg_{pollutant}'] = season_df[pollutant].mean()
                
                # Create a textual summary for embedding
                text_summary = f"Seasonal air quality data for {station_metadata.get('name', station_name)} in {station_metadata.get('city', 'Unknown City')} during {season} {year}. "
                text_summary += f"Average AQI: {seasonal_stats['avg_aqi']:.2f}, Maximum AQI: {seasonal_stats['max_aqi']:.2f}, Minimum AQI: {seasonal_stats['min_aqi']:.2f}. "
                text_summary += f"Days with AQI > 200: {seasonal_stats['days_above_200']}, Days with AQI > 300: {seasonal_stats['days_above_300']}. "
                
                # Add pollutant information
                for pollutant in pollutants:
                    if f'avg_{pollutant}' in seasonal_stats:
                        text_summary += f"Average {pollutant.upper()}: {seasonal_stats[f'avg_{pollutant}']:.2f}. "
                
                # Add location info
                if 'latitude' in station_metadata and 'longitude' in station_metadata:
                    text_summary += f"Located at coordinates: {station_metadata['latitude']}, {station_metadata['longitude']}. "
                
                # Create metadata
                metadata = {
                    'station_id': station_id,
                    'station_name': station_metadata.get('name', station_name),
                    'city': station_metadata.get('city', 'Unknown'),
                    'country': station_metadata.get('country', 'IN'),
                    'latitude': station_metadata.get('latitude'),
                    'longitude': station_metadata.get('longitude'),
                    'time_period': f"{season} {year}",
                    'period_type': 'seasonal',
                    'start_date': season_df['date'].min().strftime('%Y-%m-%d'),
                    'end_date': season_df['date'].max().strftime('%Y-%m-%d'),
                    'data_points': len(season_df),
                    **seasonal_stats
                }
                
                # Create document
                doc = Document(page_content=text_summary, metadata=metadata)
                documents.append(doc)
            
            # Create yearly documents
            yearly_data = df.groupby(df['date'].dt.year)
            for year, year_df in yearly_data:
                
                # Calculate yearly statistics
                yearly_stats = {
                    'avg_aqi': year_df['aqi_cal'].mean(),
                    'max_aqi': year_df['aqi_cal'].max(),
                    'min_aqi': year_df['aqi_cal'].min(),
                    'days_above_200': (year_df['aqi_cal'] > 200).sum(),
                    'days_above_300': (year_df['aqi_cal'] > 300).sum()
                }
                
                # Calculate pollutant averages
                for pollutant in pollutants:
                    if pollutant in year_df.columns:
                        yearly_stats[f'avg_{pollutant}'] = year_df[pollutant].mean()
                
                # Create a textual summary for embedding
                text_summary = f"Yearly air quality data for {station_metadata.get('name', station_name)} in {station_metadata.get('city', 'Unknown City')} for {year}. "
                text_summary += f"Average AQI: {yearly_stats['avg_aqi']:.2f}, Maximum AQI: {yearly_stats['max_aqi']:.2f}, Minimum AQI: {yearly_stats['min_aqi']:.2f}. "
                text_summary += f"Days with AQI > 200: {yearly_stats['days_above_200']}, Days with AQI > 300: {yearly_stats['days_above_300']}. "
                
                # Add pollutant information
                for pollutant in pollutants:
                    if f'avg_{pollutant}' in yearly_stats:
                        text_summary += f"Average {pollutant.upper()}: {yearly_stats[f'avg_{pollutant}']:.2f}. "
                
                # Add location info
                if 'latitude' in station_metadata and 'longitude' in station_metadata:
                    text_summary += f"Located at coordinates: {station_metadata['latitude']}, {station_metadata['longitude']}. "
                
                # Create metadata
                metadata = {
                    'station_id': station_id,
                    'station_name': station_metadata.get('name', station_name),
                    'city': station_metadata.get('city', 'Unknown'),
                    'country': station_metadata.get('country', 'IN'),
                    'latitude': station_metadata.get('latitude'),
                    'longitude': station_metadata.get('longitude'),
                    'time_period': str(year),
                    'period_type': 'yearly',
                    'start_date': year_df['date'].min().strftime('%Y-%m-%d'),
                    'end_date': year_df['date'].max().strftime('%Y-%m-%d'),
                    'data_points': len(year_df),
                    **yearly_stats
                }
                
                # Create document
                doc = Document(page_content=text_summary, metadata=metadata)
                documents.append(doc)
            
            # Create special documents for extreme events (very high AQI days)
            extreme_days = df[df['aqi_cal'] > 350].copy()
            for _, row in extreme_days.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d')
                
                # Create extreme event metadata
                event_stats = {
                    'aqi': row['aqi_cal']
                }
                
                # Add pollutant values
                for pollutant in pollutants:
                    if pollutant in row and not pd.isna(row[pollutant]):
                        event_stats[pollutant] = row[pollutant]
                
                # Create a textual summary
                text_summary = f"Extreme air pollution event at {station_metadata.get('name', station_name)} in {station_metadata.get('city', 'Unknown City')} on {date_str}. "
                text_summary += f"AQI reached a dangerous level of {row['aqi_cal']:.2f}. "
                
                # Add pollutant information
                for pollutant in pollutants:
                    if pollutant in row and not pd.isna(row[pollutant]):
                        text_summary += f"{pollutant.upper()}: {row[pollutant]:.2f}. "
                
                # Add location info
                if 'latitude' in station_metadata and 'longitude' in station_metadata:
                    text_summary += f"Located at coordinates: {station_metadata['latitude']}, {station_metadata['longitude']}. "
                
                # Create metadata
                metadata = {
                    'station_id': station_id,
                    'station_name': station_metadata.get('name', station_name),
                    'city': station_metadata.get('city', 'Unknown'),
                    'country': station_metadata.get('country', 'IN'),
                    'latitude': station_metadata.get('latitude'),
                    'longitude': station_metadata.get('longitude'),
                    'time_period': date_str,
                    'period_type': 'extreme_event',
                    'event_type': 'high_pollution',
                    **event_stats
                }
                
                # Create document
                doc = Document(page_content=text_summary, metadata=metadata)
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []
    
    def create_city_region_summaries(self, documents):
        """Create summary documents for cities and regions"""
        logger.info("Creating city and region summaries...")
        
        # Group by city
        city_docs = {}
        for doc in documents:
            city = doc.metadata.get('city', 'Unknown')
            if city not in city_docs:
                city_docs[city] = []
            city_docs[city].append(doc)
        
        city_summaries = []
        for city, docs in city_docs.items():
            # Get yearly docs for comparison
            yearly_docs = [d for d in docs if d.metadata.get('period_type') == 'yearly']
            if not yearly_docs:
                continue
                
            # Sort by year
            yearly_docs.sort(key=lambda x: x.metadata.get('time_period'))
            
            # Create summary text
            text_summary = f"Air quality summary for {city}, India across multiple years. "
            
            # Calculate trends
            if len(yearly_docs) > 1:
                first_year = yearly_docs[0].metadata
                last_year = yearly_docs[-1].metadata
                
                aqi_change = last_year.get('avg_aqi', 0) - first_year.get('avg_aqi', 0)
                aqi_percent = (aqi_change / first_year.get('avg_aqi', 1)) * 100 if first_year.get('avg_aqi') else 0
                
                text_summary += f"From {first_year.get('time_period')} to {last_year.get('time_period')}, "
                
                if aqi_change > 0:
                    text_summary += f"the average AQI increased by {abs(aqi_change):.2f} ({abs(aqi_percent):.1f}%), indicating worsening air quality. "
                else:
                    text_summary += f"the average AQI decreased by {abs(aqi_change):.2f} ({abs(aqi_percent):.1f}%), indicating improving air quality. "
                
                # Add pollutant trends if available
                pollutants = ['pm2_5', 'pm10', 'no2', 'so2', 'o3']
                for pollutant in pollutants:
                    poll_key = f'avg_{pollutant}'
                    if poll_key in first_year and poll_key in last_year:
                        poll_change = last_year[poll_key] - first_year[poll_key]
                        poll_percent = (poll_change / first_year[poll_key]) * 100 if first_year[poll_key] else 0
                        
                        if abs(poll_percent) > 10:  # Only mention significant changes
                            if poll_change > 0:
                                text_summary += f"{pollutant.upper()} levels increased by {abs(poll_percent):.1f}%. "
                            else:
                                text_summary += f"{pollutant.upper()} levels decreased by {abs(poll_percent):.1f}%. "
            
            # Add station count
            station_ids = set([d.metadata.get('station_id') for d in docs if 'station_id' in d.metadata])
            text_summary += f"This data is based on {len(station_ids)} air quality monitoring stations in {city}. "
            
            # Add season info if available
            seasonal_docs = [d for d in docs if d.metadata.get('period_type') == 'seasonal']
            if seasonal_docs:
                seasons = {}
                for doc in seasonal_docs:
                    season = doc.metadata.get('time_period', '').split()[0]
                    if season not in seasons:
                        seasons[season] = []
                    seasons[season].append(doc.metadata.get('avg_aqi', 0))
                
                # Find worst season
                season_avgs = {s: sum(vals)/len(vals) for s, vals in seasons.items() if vals}
                if season_avgs:
                    worst_season = max(season_avgs, key=season_avgs.get)
                    text_summary += f"Air quality is typically worst during the {worst_season} season. "
            
            # Create metadata
            metadata = {
                'city': city,
                'country': 'IN',
                'summary_type': 'city',
                'stations_count': len(station_ids),
                'years_covered': [d.metadata.get('time_period') for d in yearly_docs],
                'latest_year_avg_aqi': yearly_docs[-1].metadata.get('avg_aqi') if yearly_docs else None
            }
            
            # Create document
            doc = Document(page_content=text_summary, metadata=metadata)
            city_summaries.append(doc)
        
        return city_summaries
    
    def create_vector_store(self, documents):
        """Create FAISS vector store from documents"""
        logger.info(f"Creating vector store with {len(documents)} documents...")
        
        # Create FAISS vector store
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model  # Now using the compatible wrapper class
        )
        
        # Save the vector store
        vectorstore.save_local(self.output_folder)
        logger.info(f"Vector store saved to {self.output_folder}")
        
        return vectorstore
    
    def create_metadata_index(self, documents):
        """Create a metadata index for efficient filtering"""
        logger.info("Creating metadata index...")
        
        # Extract city names
        cities = sorted(list(set([doc.metadata.get('city', 'Unknown') 
                              for doc in documents 
                              if doc.metadata.get('city') is not None])))
        
        # Extract station names
        stations = {}
        for doc in documents:
            city = doc.metadata.get('city')
            station_id = doc.metadata.get('station_id')
            station_name = doc.metadata.get('station_name')
            
            if city and station_id and station_name:
                if city not in stations:
                    stations[city] = []
                if {'id': station_id, 'name': station_name} not in stations[city]:
                    stations[city].append({'id': station_id, 'name': station_name})
        
        # Extract years
        years = sorted(list(set([doc.metadata.get('time_period') 
                             for doc in documents 
                             if doc.metadata.get('period_type') == 'yearly'])))
        
        # Create index
        metadata_index = {
            'cities': cities,
            'stations': stations,
            'years': years,
            'total_documents': len(documents)
        }
        
         # Convert any NumPy integers to Python integers
        metadata_index = self._convert_np_int64(metadata_index)
        
        # Save to file
        with open(os.path.join(self.output_folder, 'metadata_index.json'), 'w') as f:
            json.dump(metadata_index, f, indent=2)
        
        logger.info(f"Metadata index saved with {len(cities)} cities and {len(years)} years")
        
        return metadata_index

    def _convert_np_int64(self, obj):
        """Recursively convert numpy.int64 to int"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_np_int64(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_np_int64(item) for item in obj]
        return obj
        
    def build(self):
        """Build the vector database"""
        # Load metadata
        self.load_metadata()
        
        # Load and process all data
        documents = self.load_and_process_data()
        logger.info(f"Processed {len(documents)} documents from station data")
        
        # Create city summaries
        city_summaries = self.create_city_region_summaries(documents)
        logger.info(f"Created {len(city_summaries)} city/region summaries")
        
        # Combine all documents
        all_documents = documents + city_summaries
        logger.info(f"Total documents: {len(all_documents)}")
        
        # Create vector store
        vectorstore = self.create_vector_store(all_documents)
        
        # Create metadata index
        metadata_index = self.create_metadata_index(all_documents)
        
        return vectorstore, metadata_index

# If handling a large dataset, you might want to process in smaller chunks
def build_vector_db_in_chunks(data_folder, metadata_file, output_folder, chunk_size=20000):
    """Build vector database in chunks to manage memory usage"""
    logger.info("Starting vector database build in chunks...")
    
    # Create the database object
    vector_db = AQIVectorDatabase(data_folder, metadata_file, output_folder)
    vector_db.load_metadata()
    
    # Load all documents
    all_docs = vector_db.load_and_process_data()
    logger.info(f"Processed {len(all_docs)} documents from station data")
    
    # Create city summaries
    city_summaries = vector_db.create_city_region_summaries(all_docs)
    logger.info(f"Created {len(city_summaries)} city/region summaries")
    
    # Combine all documents
    all_documents = all_docs + city_summaries
    logger.info(f"Total documents: {len(all_documents)}")
    
    # Process in chunks
    chunk_count = len(all_documents) // chunk_size + (1 if len(all_documents) % chunk_size > 0 else 0)
    logger.info(f"Processing in {chunk_count} chunks of {chunk_size} documents")
    
    faiss_index = None
    
    for i in range(chunk_count):
        start_idx = i * chunk_size
        end_idx = min((i+1) * chunk_size, len(all_documents))
        chunk_docs = all_documents[start_idx:end_idx]
        
        logger.info(f"Processing chunk {i+1}/{chunk_count} with {len(chunk_docs)} documents")
        
        # Create or update vector store
        if i == 0:
            # First chunk - create the vector store
            faiss_index = FAISS.from_documents(
                documents=chunk_docs,
                embedding=vector_db.embedding_model
            )
        else:
            # Subsequent chunks - add to existing index
            tmp_index = FAISS.from_documents(
                documents=chunk_docs,
                embedding=vector_db.embedding_model
            )
            faiss_index.merge_from(tmp_index)
        
        logger.info(f"Processed chunk {i+1}/{chunk_count}")
    
    # Save the final vector store
    faiss_index.save_local(output_folder)
    logger.info(f"Vector store saved to {output_folder}")
    
    # Create metadata index
    metadata_index = vector_db.create_metadata_index(all_documents)
    
    return faiss_index, metadata_index

# Example usage
if __name__ == "__main__":
    # Paths to data
    data_folder = "data/aqi_stations"
    metadata_file = "data/station_metadata.csv"
    output_folder = "vector_db"
    
    # Choose one of these approaches based on your dataset size and memory constraints
    
    # Option 1: Standard build (good for datasets that fit in memory)
    # vector_db = AQIVectorDatabase(data_folder, metadata_file, output_folder)
    # vectorstore, metadata = vector_db.build()
    
    # Option 2: Chunked build (better for very large datasets)
    vectorstore, metadata = build_vector_db_in_chunks(data_folder, metadata_file, output_folder, chunk_size=10000)
    
    # Example query
    query = "What was the air quality like in Delhi during winter 2023?"
    docs = vectorstore.similarity_search(query, k=3)
    
    for doc in docs:
        print("---")
        print(doc.page_content)
        print("---")