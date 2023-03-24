import React from 'react';
import { Routes, Route } from 'react-router-dom';

import DashboardPage from "./pages/dashboard-page.component";

import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

function App() {
  return (
    <div>
        <link
            rel="stylesheet"
            href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css"
            integrity="sha384-rbsA2VBKQhggwzxH7pPCaAqO46MgnOM80zW1RWuH61DGLwZJEdK2Kadq2F9CUG65"
            crossOrigin="anonymous"
        />
        <Routes>
            <Route path='/' element={<DashboardPage />}/>
        </Routes>
    </div>
  );
}

export default App;
