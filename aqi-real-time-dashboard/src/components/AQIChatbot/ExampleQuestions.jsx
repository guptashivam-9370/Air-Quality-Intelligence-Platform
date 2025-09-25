import React from 'react';
import { Box, Typography, Button, List, ListItem } from '@mui/material';

function ExampleQuestions({ onSelectQuestion, isDisabled, isDarkMode }) {
  const exampleQuestions = [
    "What was the average PM2.5 level in Delhi during winter 2023?",
    "Which city had the worst air quality in 2022?",
    "How does Mumbai's air quality compare to Chennai?",
    "Show me trends in PM10 and PM2.5 levels in Bengaluru from 2020 to 2024",
    "What are the seasonal patterns of air pollution in Kolkata?",
    "Which stations recorded the highest SO2 levels in 2023?",
    "How has air quality in Delhi changed over the past 5 years?",
    "What's the correlation between PM10 and PM2.5 in industrial cities?"
  ];

  // Define text color based on dark mode
  const textColor = isDarkMode ? "white" : "black";

  return (
    <Box>
      <Typography variant="h6" sx={{ mb: 2, color: textColor }}>
        Example Questions
      </Typography>
      <List>
        {exampleQuestions.map((question, index) => (
          <ListItem key={index} disablePadding sx={{ mb: 1 }}>
            <Button
              fullWidth
              variant="outlined"
              size="small"
              sx={{ 
                justifyContent: 'flex-start', 
                textAlign: 'left', 
                textTransform: 'none',
                color: textColor,
                borderColor: isDarkMode ? 'rgba(255, 255, 255, 0.23)' : 'rgba(0, 0, 0, 0.23)',
                '&:hover': {
                  borderColor: isDarkMode ? 'rgba(255, 255, 255, 0.5)' : 'rgba(0, 0, 0, 0.5)'
                }
              }}
              onClick={() => onSelectQuestion(question)}
              disabled={isDisabled}
            >
              {question}
            </Button>
          </ListItem>
        ))}
      </List>
    </Box>
  );
}

export default ExampleQuestions;