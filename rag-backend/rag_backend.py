import os
import re
import json
from typing import Dict, List, Optional, Any, Tuple
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Create a wrapper class for SentenceTransformer
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with model name"""
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using SentenceTransformer"""
        return self.model.encode(texts, convert_to_numpy=True).tolist()
        
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using SentenceTransformer"""
        return self.model.encode(text, convert_to_numpy=True).tolist()

class AQIRagSystem:
    # Default AQI categories based on EPA standards
    DEFAULT_AQI_CATEGORIES = {
        "pm2_5": [
            {"min": 0, "max": 12, "category": "Good"},
            {"min": 12.1, "max": 35.4, "category": "Moderate"},
            {"min": 35.5, "max": 55.4, "category": "Unhealthy for Sensitive Groups"},
            {"min": 55.5, "max": 150.4, "category": "Unhealthy"},
            {"min": 150.5, "max": 250.4, "category": "Very Unhealthy"},
            {"min": 250.5, "max": float('inf'), "category": "Hazardous"}
        ],
        "pm10": [
            {"min": 0, "max": 54, "category": "Good"},
            {"min": 55, "max": 154, "category": "Moderate"},
            {"min": 155, "max": 254, "category": "Unhealthy for Sensitive Groups"},
            {"min": 255, "max": 354, "category": "Unhealthy"},
            {"min": 355, "max": 424, "category": "Very Unhealthy"},
            {"min": 425, "max": float('inf'), "category": "Hazardous"}
        ]
    }
    
    # WHO Guidelines (μg/m³)
    DEFAULT_WHO_GUIDELINES = {
        "pm2_5": {"annual": 10, "24_hour": 25},
        "pm10": {"annual": 20, "24_hour": 50},
        "no2": {"annual": 40, "1_hour": 200},
        "so2": {"24_hour": 40, "10_minute": 500},
        "o3": {"8_hour": 100},
        "co": {"24_hour": 4000, "8_hour": 10000}
    }
    
    def __init__(
        self, 
        vector_db_path: str = "./vector_db",
        api_key: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gemini-1.5-flash",
        llm_temperature: float = 0.2,
        llm_top_p: float = 0.95,
        llm_max_tokens: int = 1024,
        retrieval_k: int = 8,
        retrieval_fetch_k: int = 20,
        retrieval_lambda: float = 0.7,
        monitoring_stations: int = 662,
        data_start_year: str = "2020",
        data_end_year: str = "2025",
        verbose: bool = False,
        aqi_categories: Optional[Dict] = None,
        who_guidelines: Optional[Dict] = None
    ):
        """
        Initialize AQI RAG System with configurable parameters
        
        Args:
            vector_db_path: Path to vector database
            api_key: Google API key
            embedding_model: Name of the embedding model
            llm_model: Name of the LLM model
            llm_temperature: Temperature for LLM
            llm_top_p: Top-p for LLM
            llm_max_tokens: Max output tokens for LLM
            retrieval_k: Number of documents to retrieve
            retrieval_fetch_k: Number of candidates to consider
            retrieval_lambda: Lambda for MMR diversity
            monitoring_stations: Number of monitoring stations
            data_start_year: Start year of data
            data_end_year: End year of data
            verbose: Enable verbose logging
            aqi_categories: Custom AQI categories (uses defaults if None)
            who_guidelines: Custom WHO guidelines (uses defaults if None)
        """
        # Store parameters
        self.vector_db_path = vector_db_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_top_p = llm_top_p
        self.llm_max_tokens = llm_max_tokens
        self.retrieval_k = retrieval_k
        self.retrieval_fetch_k = retrieval_fetch_k
        self.retrieval_lambda = retrieval_lambda
        self.monitoring_stations = monitoring_stations
        self.data_start_year = data_start_year
        self.data_end_year = data_end_year
        self.verbose = verbose
        self.aqi_categories = aqi_categories or self.DEFAULT_AQI_CATEGORIES
        self.who_guidelines = who_guidelines or self.DEFAULT_WHO_GUIDELINES
        
        # Check for API key
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass as parameter.")
        
        # Initialize embedding model
        self.embeddings = SentenceTransformerEmbeddings(model_name=self.embedding_model)
        
        # Load vector database
        self.vector_store = FAISS.load_local(
            self.vector_db_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Initialize the retriever with improved settings
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.retrieval_k,
                "fetch_k": self.retrieval_fetch_k,
                "lambda_mult": self.retrieval_lambda
            }
        )
        
        # Initialize Google Gemini model
        self.llm = GoogleGenerativeAI(
            model=self.llm_model,
            google_api_key=self.api_key,
            temperature=self.llm_temperature,
            top_p=self.llm_top_p,
            max_output_tokens=self.llm_max_tokens
        )
        
        # Create RAG chain
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            verbose=self.verbose
        )
    
    def get_retriever_for_query(self, query: str) -> Any:
        """Get a customized retriever based on query content"""
        # Load metadata index if exists
        metadata_path = os.path.join(self.vector_db_path, "metadata_index.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata_index = json.load(f)
                available_cities = metadata_index.get("cities", [])
                available_years = metadata_index.get("years", [])
        else:
            # Fallback to default years range
            available_cities = []  # Will be empty if no metadata
            available_years = [str(year) for year in range(int(self.data_start_year), int(self.data_end_year) + 1)]
        
        # Time-related terms
        time_terms = {
            "seasons": ["winter", "summer", "monsoon", "spring", "autumn", "fall"],
            "months": ["january", "february", "march", "april", "may", "june", 
                      "july", "august", "september", "october", "november", "december"],
            "years": available_years
        }
        
        # Flatten time terms for matching
        all_time_terms = []
        for terms in time_terms.values():
            all_time_terms.extend(terms)
        
        # Default retriever configuration
        retriever_config = {
            "search_type": "mmr",
            "search_kwargs": {
                "k": self.retrieval_k,
                "fetch_k": self.retrieval_fetch_k,
                "lambda_mult": self.retrieval_lambda
            }
        }
        
        # Check for city and time matches in query
        query_lower = query.lower()
        city_matches = [city for city in available_cities if city.lower() in query_lower]
        time_matches = [term for term in all_time_terms if term.lower() in query_lower]
        
        # Add metadata filtering if specific city and time are mentioned
        if city_matches and time_matches:
            filter_dict = {"city": city_matches[0].title()}
            retriever_config["search_type"] = "similarity"
            retriever_config["search_kwargs"]["k"] = min(self.retrieval_k + 2, 15)
            retriever_config["search_kwargs"]["filter"] = filter_dict
        
        return self.vector_store.as_retriever(**retriever_config)
    
    def handle_comparison_query(self, query: str, docs: List[Any]) -> str:
        """Handle queries that ask for comparisons between cities or time periods"""
        comparison_keywords = ["compare", "comparison", "versus", "vs", "difference between", "better than", "worse than"]
        if not any(keyword in query.lower() for keyword in comparison_keywords):
            return ""
            
        # Group documents by city
        city_docs = {}
        for doc in docs:
            city = doc.metadata.get("city", "Unknown")
            if city not in city_docs:
                city_docs[city] = []
            city_docs[city].append(doc)
        
        # If we have multiple cities, create a comparison table
        if len(city_docs) > 1:
            comparison_text = "\n\n## Air Quality Comparison\n\n"
            comparison_text += "| City | Avg AQI | PM2.5 (μg/m³) | PM10 (μg/m³) | NO2 (μg/m³) | SO2 (μg/m³) | O3 (μg/m³) |\n"
            comparison_text += "|------|---------|---------------|--------------|-------------|-------------|------------|\n"
            
            city_stats = []
            for city, city_docs_list in city_docs.items():
                # Calculate averages from metadata
                stats = {
                    "city": city,
                    "avg_aqi": self._safe_average([d.metadata.get("avg_aqi", 0) for d in city_docs_list]),
                    "avg_pm2_5": self._safe_average([d.metadata.get("avg_pm2_5", 0) for d in city_docs_list]),
                    "avg_pm10": self._safe_average([d.metadata.get("avg_pm10", 0) for d in city_docs_list]),
                    "avg_no2": self._safe_average([d.metadata.get("avg_no2", 0) for d in city_docs_list]),
                    "avg_so2": self._safe_average([d.metadata.get("avg_so2", 0) for d in city_docs_list]),
                    "avg_o3": self._safe_average([d.metadata.get("avg_o3", 0) for d in city_docs_list])
                }
                city_stats.append(stats)
            
            # Sort by AQI (best to worst)
            city_stats.sort(key=lambda x: x["avg_aqi"])
            
            # Add rows to table
            for stats in city_stats:
                comparison_text += f"| {stats['city']} | {stats['avg_aqi']:.1f} | {stats['avg_pm2_5']:.1f} | "
                comparison_text += f"{stats['avg_pm10']:.1f} | {stats['avg_no2']:.1f} | "
                comparison_text += f"{stats['avg_so2']:.1f} | {stats['avg_o3']:.1f} |\n"
            
            # Add summary
            if city_stats:
                comparison_text += f"\n**Best Air Quality**: {city_stats[0]['city']} (AQI: {city_stats[0]['avg_aqi']:.1f})\n"
                comparison_text += f"**Worst Air Quality**: {city_stats[-1]['city']} (AQI: {city_stats[-1]['avg_aqi']:.1f})\n"
            
            return comparison_text
        
        return ""
    
    def _safe_average(self, values: List[Any]) -> float:
        """Calculate average safely, handling non-numeric values"""
        numeric_values = []
        for v in values:
            try:
                val = float(v)
                if val > 0:  # Ignore zero/negative values
                    numeric_values.append(val)
            except (TypeError, ValueError):
                continue
        return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
    
    def _get_aqi_category(self, value: float, pollutant: str) -> str:
        """Convert raw pollutant values to health categories"""
        if pollutant not in self.aqi_categories:
            return ""
        
        categories = self.aqi_categories[pollutant]
        for category_info in categories:
            if category_info["min"] <= value <= category_info["max"]:
                return category_info["category"]
        
        # If value exceeds all ranges, return the last category (typically "Hazardous")
        return categories[-1]["category"] if categories else ""
    
    def post_process_answer(self, answer: str, sources: List[Dict[str, Any]]) -> str:
        """Clean and enhance the generated answer"""
        # Remove common hallucination phrases
        answer = re.sub(r"I don't have specific information about.*?in the provided context", 
                      "Based on the available data:", answer)
        
        # Convert raw numbers to more readable format with health context
        answer = re.sub(r"PM2\.5: (\d+\.?\d*)", 
                      lambda m: f"PM2.5: {float(m.group(1)):.1f} μg/m³ ({self._get_aqi_category(float(m.group(1)), 'pm2_5')})", 
                      answer)
        
        answer = re.sub(r"PM10: (\d+\.?\d*)", 
                      lambda m: f"PM10: {float(m.group(1)):.1f} μg/m³ ({self._get_aqi_category(float(m.group(1)), 'pm10')})", 
                      answer)
        
        # Add source attribution if missing
        if len(sources) > 0 and "according to the data" not in answer.lower():
            answer += "\n\nThis analysis is based on data from monitoring stations in "
            cities = set([s["city"] for s in sources if s["city"] != "Unknown"])
            answer += ", ".join(cities) + "."
        
        return answer
    
    def _generate_prompt(self, user_query: str, context: str) -> str:
        """Generate a dynamic prompt based on configuration and query type"""
        # Format WHO guidelines
        who_guidelines_text = ""
        for pollutant, limits in self.who_guidelines.items():
            pollutant_name = pollutant.upper().replace("_", ".")
            limit_parts = []
            for period, value in limits.items():
                period_formatted = period.replace("_", "-")
                limit_parts.append(f"{value} μg/m³ ({period_formatted})")
            who_guidelines_text += f"    * {pollutant_name}: {', '.join(limit_parts)}\n"
        
        # Detect query type
        query_lower = user_query.lower()
        is_comparison = any(word in query_lower for word in ["compare", "comparison", "versus", "vs", "difference between"])
        is_trend = any(word in query_lower for word in ["trend", "change", "increase", "decrease", "over time"])
        is_health = any(word in query_lower for word in ["health", "impact", "safe", "dangerous", "risk"])
        
        # Build dynamic prompt
        prompt = f"""You are an expert air quality data analyst assistant specializing in Indian air quality monitoring data.

SYSTEM CONTEXT:
- Data Source: {self.monitoring_stations} monitoring stations across India
- Time Period: {self.data_start_year} to {self.data_end_year}
- Analysis Type: {'Comparison' if is_comparison else 'Trend Analysis' if is_trend else 'Health Impact' if is_health else 'General Analysis'}

RETRIEVED DATA:
{context}

USER QUERY: {user_query}

ANALYSIS GUIDELINES:

1. Query Understanding:
   - First determine if this is an air quality related query
   - For non-AQI queries, provide a brief, helpful response without using the context
   - For AQI queries, proceed with detailed analysis

2. Response Structure:
   - Start with a direct answer to the main question
   - Support with specific data points from the context
   - Include relevant statistics (averages, ranges, percentages)
   - Mention temporal and spatial patterns when relevant

3. Data Interpretation:
   - Compare values to WHO air quality guidelines:
{who_guidelines_text}
   - Explain health implications using standard AQI categories
   - Provide practical recommendations based on air quality levels

4. Quality Standards:
   - Use clear, conversational language
   - Organize information logically
   - Highlight key findings
   - Keep technical terms accessible

5. Data Integrity:
   - Use ONLY information from the provided context
   - Clearly state if requested data is not available
   - Never fabricate or extrapolate beyond given data
   - Maintain scientific accuracy

"""
        
        # Add specific instructions based on query type
        if is_comparison:
            prompt += """
6. Comparison Analysis:
   - Structure response as a clear comparison
   - Use parallel structure for entities being compared
   - Include quantitative differences
   - Highlight which performs better/worse and why
"""
        elif is_trend:
            prompt += """
6. Trend Analysis:
   - Identify overall trend direction and magnitude
   - Quantify rate of change
   - Discuss seasonal patterns if evident
   - Suggest evidence-based causes for trends
"""
        elif is_health:
            prompt += """
6. Health Impact Focus:
   - Translate AQI/pollutant levels into health impacts
   - Identify vulnerable populations
   - Provide specific health recommendations
   - Distinguish short-term vs long-term effects
"""
        
        prompt += "\nRESPONSE:"
        
        return prompt
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """Process a user query and return the response with sources"""
        try:
            # Get a query-specific retriever
            retriever = self.get_retriever_for_query(user_query)
            
            # Get relevant documents
            docs = retriever.get_relevant_documents(user_query)
            
            if not docs:
                return {
                    "answer": "I don't have sufficient information about this in my database. Please try a different query about air quality in India.",
                    "sources": []
                }
            
            # Extract relevant context
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
            
            # Generate dynamic prompt
            formatted_query = self._generate_prompt(user_query, context)
                            
            # Get response from LLM
            response = self.llm.invoke(formatted_query)
            
            # Check for comparison queries and add table if needed
            comparison_text = self.handle_comparison_query(user_query, docs)
            if comparison_text:
                response += comparison_text
            
            # Extract sources
            sources = []
            for doc in docs:
                source_info = {
                    "content_preview": doc.page_content[:150] + "...",
                    "station_name": doc.metadata.get("station_name", "Unknown"),
                    "city": doc.metadata.get("city", "Unknown"),
                    "time_period": doc.metadata.get("time_period", "Unknown"),
                    "doc_type": doc.metadata.get("period_type", "Unknown")
                }
                sources.append(source_info)
           
            # Post-process the answer
            answer = self.post_process_answer(response, sources)
            
            return {
                "answer": answer,
                "sources": sources
            }
        
        except Exception as e:
            import traceback
            error_details = f"Error processing your query: {str(e)}"
            if self.verbose:
                error_details += f"\n\nTraceback:\n{traceback.format_exc()}"
            return {
                "answer": error_details,
                "sources": []
            }

    def get_available_metadata(self) -> Dict[str, List[str]]:
        """
        Get available cities and years from the vector store metadata
        
        Returns:
            Dictionary with 'cities' and 'years' lists
        """
        metadata_path = os.path.join(self.vector_db_path, "metadata_index.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"Error loading metadata index: {e}")
        
        # Return empty metadata if file doesn't exist
        return {
            "cities": [],
            "years": [str(year) for year in range(int(self.data_start_year), int(self.data_end_year) + 1)],
            "total_documents": 0
        }