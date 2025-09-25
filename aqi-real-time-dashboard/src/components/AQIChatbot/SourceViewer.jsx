import React from 'react';
import { Box, Typography, Divider, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';

function SourceViewer({ sources, viewType }) {
  if (!sources || sources.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary">
        No source information available
      </Typography>
    );
  }

  if (viewType === 'table') {
    return (
      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Station Name</TableCell>
              <TableCell>City</TableCell>
              <TableCell>Time Period</TableCell>
              <TableCell>Document Type</TableCell>
              <TableCell>Preview</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {sources.map((source, index) => (
              <TableRow key={index}>
                <TableCell>{source.station_name || 'N/A'}</TableCell>
                <TableCell>{source.city || 'N/A'}</TableCell>
                <TableCell>{source.time_period || 'N/A'}</TableCell>
                <TableCell>{source.doc_type || 'N/A'}</TableCell>
                <TableCell sx={{ maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {source.content_preview || 'N/A'}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  }

  // Detailed view
  return (
    <Box>
      {sources.map((source, index) => (
        <Box key={index} sx={{ mb: 3 }}>
          <Typography variant="h6">
            Source {index + 1}: {source.station_name} - {source.city} ({source.time_period})
          </Typography>
          <Typography variant="body1" fontWeight="bold" sx={{ mt: 1 }}>
            Document Type: {source.doc_type || 'Unknown'}
          </Typography>
          <Paper sx={{ p: 2, mt: 1, backgroundColor: '#f0f2f6', borderRadius: 1 }}>
            <Typography variant="body2">
              {source.content_preview || 'No preview available'}
            </Typography>
          </Paper>
          {index < sources.length - 1 && <Divider sx={{ my: 2 }} />}
        </Box>
      ))}
    </Box>
  );
}

export default SourceViewer;