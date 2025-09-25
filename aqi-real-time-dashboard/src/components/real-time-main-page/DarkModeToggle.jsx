// DarkModeToggle.jsx
import React from 'react';

const DarkModeToggle = ({ isDark, toggleDarkMode }) => {
  return (
    <button
      onClick={toggleDarkMode}
      className="p-2 bg-gray-200 dark:bg-gray-700 rounded-full focus:outline-none transition-colors"
    >
      {isDark ? "Light Mode" : "Dark Mode"}
    </button>
  );
};

export default DarkModeToggle;
