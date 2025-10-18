import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx'; // Imports the main component

// Mount the React App component into the HTML element with id="root"
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);