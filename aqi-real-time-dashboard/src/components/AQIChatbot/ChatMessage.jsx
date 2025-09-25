import React, { useState } from 'react';
import { Box, Typography, Avatar, Paper, Accordion, AccordionSummary, AccordionDetails, Tab, Tabs } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ReactMarkdown from 'react-markdown';
import Plot from 'react-plotly.js';
import PersonIcon from '@mui/icons-material/Person';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import SourceViewer from './SourceViewer';

function ChatMessage({ message, queryData }) {
  const [sourceTab, setSourceTab] = useState(0);
  
  const handleSourceTabChange = (event, newValue) => {
    setSourceTab(newValue);
  };

  // Function to create visualization if data exists
  const renderVisualization = () => {
    const data = queryData?.visualizationData;
    if (!data || data.length === 0) return null;

    // Group by pollutant
    const pollutants = Array.from(new Set(data.map(item => item.Pollutant)));
    const plotData = pollutants.map(pollutant => {
      const values = data
        .filter(item => item.Pollutant === pollutant)
        .map(item => item.Value);
      
      return {
        x: [pollutant],
        y: values.length > 1 ? [values.reduce((a, b) => a + b, 0) / values.length] : values,
        type: 'bar',
        name: pollutant
      };
    });

    return (
      <Accordion sx={{ mt: 2, backgroundColor: '#f9f9f9' }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>View Visualization</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Plot
            data={plotData}
            layout={{
              title: 'Extracted Pollutant Values',
              autosize: true,
              height: 400,
              yaxis: { title: 'Value (μg/m³)' }
            }}
            style={{ width: '100%' }}
            useResizeHandler={true}
          />
        </AccordionDetails>
      </Accordion>
    );
  };

  // Function to render sources if they exist
  const renderSources = () => {
    const sources = queryData?.sources;
    if (!sources || sources.length === 0) return null;

    return (
      <Accordion sx={{ mt: 2, backgroundColor: '#f9f9f9' }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>View Sources ({sources.length})</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={sourceTab} onChange={handleSourceTabChange}>
              <Tab label="Detailed View" />
              <Tab label="Table View" />
            </Tabs>
          </Box>
          <Box sx={{ p: 2 }}>
            {sourceTab === 0 ? (
              <SourceViewer sources={sources} viewType="detailed" />
            ) : (
              <SourceViewer sources={sources} viewType="table" />
            )}
          </Box>
        </AccordionDetails>
      </Accordion>
    );
  };

  return (
    <Box 
      sx={{ 
        display: 'flex', 
        flexDirection: 'column',
        mb: 2
      }}
    >
      <Box 
        sx={{ 
          display: 'flex',
          alignItems: 'flex-start',
          ...(message.role === 'user' ? {
            alignSelf: 'flex-end',
            maxWidth: '80%',
          } : {
            alignSelf: 'flex-start',
            maxWidth: '80%',
          })
        }}
      >
        <Avatar 
          sx={{ 
            mr: 1, 
            bgcolor: message.role === 'user' ? '#1976d2' : '#f5f5f5',
            color: message.role === 'user' ? 'white' : '#1976d2'
          }}
        >
          {message.role === 'user' ? <PersonIcon /> : <SmartToyIcon />}
        </Avatar>
        <Paper 
          elevation={1} 
          sx={{ 
            p: 2, 
            borderRadius: 2,
            backgroundColor: message.role === 'user' ? '#e3f2fd' : '#f5f5f5',
          }}
        >
          <Typography variant="subtitle2" fontWeight="bold" color="text.secondary">
            {message.role === 'user' ? 'You' : 'AQI Assistant'}
          </Typography>
          <Box sx={{ mt: 1 }}>
            <ReactMarkdown>
              {message.content}
            </ReactMarkdown>
          </Box>
        </Paper>
      </Box>

      {/* Only render additional content for assistant messages with data */}
      {message.role === 'assistant' && queryData && (
        <Box sx={{ ml: 7 }}>
          {renderVisualization()}
          {renderSources()}
        </Box>
      )}
    </Box>
  );
}

export default ChatMessage;