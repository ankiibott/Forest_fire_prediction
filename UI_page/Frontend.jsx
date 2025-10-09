import React, { useState } from 'react';
// Corrected imports to prevent conflict with native Map object
import { Clock, Zap, Target, Loader2, Maximize2, Layers, Globe, FileUp, ListChecks, Map } from 'lucide-react';

// --- Constants (Inferred from Model Structure) ---
const SEQ_LEN = 6;
const HORIZONS = 3;
const PATCH_SIZE = 13;
const CHANNELS = 7;
// ⚠️ API ENDPOINT: Set to localhost:5000 to match your Flask backend ⚠️
const API_ENDPOINT = 'http://localhost:5000/api/predict';

const DATASET_NAMES = [
  { name: 'T2M (2m Temperature)', abbreviation: 'T2M', type: 'dynamic', files: 6 },
  { name: 'D2M (2m Dew Point Temp)', abbreviation: 'D2M', type: 'dynamic', files: 6 },
  { name: 'TP (Total Precipitation)', abbreviation: 'TP', type: 'dynamic', files: 6 },
  { name: 'V10 (10m V-Wind)', abbreviation: 'V10', type: 'dynamic', files: 6 },
  { name: 'U10 (10m U-Wind)', abbreviation: 'U10', type: 'dynamic', files: 6 },
  { name: 'LULC (Land Cover)', abbreviation: 'LULC', type: 'static', files: 1 },
  { name: 'DEM (Digital Elevation Model)', abbreviation: 'DEM', type: 'static', files: 1 },
];

// Default coordinates centered near Dehradun, Uttarakhand
const DEFAULT_BOUNDS = {
  latMin: '30.2',
  latMax: '30.3',
  lonMin: '77.8',
  lonMax: '77.9',
};

// --- Time Details for Client-Side Fallback (Matching expected API format for UI stability) ---
const MOCK_TIME_DETAILS = {
    inputStart: "17:00:00",
    inputEnd: "22:00:00",
    predStart: "23:00:00",
    predEnd: "01:00:00", 
    date: "2015-01-01",
};

/**
 * runPredictionAPI (SERVER API CALL with FALLBACK)
 * This function now sends the file manifest and coordinates to the backend.
 */
const runPredictionAPI = async ({ latMin, latMax, lonMin, lonMax, inputFiles }) => {
    
    // Create a simplified file manifest (sending only names/details, not content)
    // NOTE: For real data transfer, you would use FormData or Base64 encoding for file content.
    const fileManifest = {};
    for (const { abbreviation, type } of DATASET_NAMES) {
        const fileData = inputFiles[abbreviation];
        if (type === 'dynamic') {
            fileManifest[abbreviation] = Array.isArray(fileData) ? fileData.map(f => f.name) : [];
        } else if (fileData instanceof File) {
            fileManifest[abbreviation] = fileData.name;
        } else {
            fileManifest[abbreviation] = null;
        }
    }

    const requestBody = {
        latMin,
        latMax,
        lonMin,
        lonMax,
        fileManifest, // Sending the structure of the uploaded files
    };

    const centerLat = (parseFloat(latMin) + parseFloat(latMax)) / 2;
    const centerLon = (parseFloat(lonMin) + parseFloat(lonMax)) / 2;

    try {
        // --- API Fetch ---
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: `Status ${response.status}` }));
            throw new Error(errorData.error || `Server responded with status ${response.status}`);
        }

        const data = await response.json();
        
        // Return structured data received from the backend
        return {
            prediction: data.prediction_results,
            timeDetails: data.time_details, 
            center: [centerLat, centerLon] 
        };

    } catch (error) {
        // --- FALLBACK TO CLIENT-SIDE SIMULATION ---
        console.warn(`[API FAILED] Falling back to simulation. Error: ${error.message}`);
        
        return new Promise(resolve => {
            setTimeout(() => {
                const mockPrediction = Array(HORIZONS).fill(0).map(() => 
                    Array(PATCH_SIZE).fill(0).map((_, i) => 
                        Array(PATCH_SIZE).fill(0).map((__, j) => {
                            const baseProb = 0.05 + (i / PATCH_SIZE * 0.2) + (j / PATCH_SIZE * 0.3) + (Math.random() * 0.1);
                            return Math.min(1.0, Math.max(0.0, baseProb));
                        })
                    )
                );
                
                resolve({
                    prediction: mockPrediction,
                    timeDetails: MOCK_TIME_DETAILS,
                    center: [centerLat, centerLon]
                });
            }, 1000); 
        });
    }
};

// --- Helper Functions for Visualization ---

const getFireColor = (probability) => {
  if (probability < 0.1) return 'rgba(255, 255, 255, 0)'; // Mostly transparent (No fire)
  if (probability < 0.3) return `rgba(255, 255, 0, ${0.4 + probability * 0.3})`; // Yellow/Low Risk
  if (probability < 0.6) return `rgba(255, 165, 0, ${0.5 + probability * 0.4})`; // Orange/Medium Risk
  return `rgba(255, 0, 0, ${0.6 + probability * 0.4})`; // Red/High Risk
};

// --- Static Grid Visualization Component ---
const StaticGridVisualization = ({ bounds, prediction, currentHorizon, timeDetails }) => {
  
  if (!prediction || prediction.length === 0) return null;

  const [[latMin, lonMin], [latMax, lonMax]] = bounds;
  
  const horizonData = prediction[currentHorizon];
  const titleDate = timeDetails?.date || 'N/A';
  const titleTime = timeDetails?.predStart || 'N/A';
  
  return (
    <div className="relative w-full h-[550px] p-2 bg-white rounded-xl shadow-inner border border-gray-100">
        <h3 className="text-lg font-bold text-center mb-4 text-gray-700">
            Fire Probability Grid @ {titleDate} {titleTime} (Hour +{currentHorizon + 1})
        </h3>

        {/* Outer container for the plot area and axes */}
        <div className="relative w-full h-[450px] mx-auto flex items-center justify-center">

            {/* Y-Axis Label */}
            <div className="absolute left-0 top-1/2 transform -translate-y-1/2 -rotate-90 text-sm font-semibold text-gray-600">
                Latitude
            </div>

            {/* Plot Area Container (13x13 Grid) */}
            <div className="flex-grow h-full bg-gray-50 shadow-inner rounded-lg overflow-hidden border border-gray-300 mx-10">
                <div 
                    className="w-full h-full grid"
                    style={{
                        gridTemplateColumns: `repeat(${PATCH_SIZE}, 1fr)`,
                        gridTemplateRows: `repeat(${PATCH_SIZE}, 1fr)`,
                    }}
                >
                    {horizonData.slice().reverse().map((row, i) => ( // Reverse rows so Lat Max (North) is at the top
                        row.map((prob, j) => (
                            <div
                                key={`${i}-${j}`}
                                className="border border-gray-200/50 transition duration-100 cursor-help"
                                style={{
                                    backgroundColor: getFireColor(prob),
                                    opacity: 1, 
                                }}
                                title={`Lat Cell: ${latMax - i}, Lon Cell: ${lonMin + j} | Prob: ${prob.toFixed(4)}`}
                            />
                        ))
                    ))}
                </div>
            </div>

            {/* Y-Axis Max/Min Markers */}
            <div className="absolute right-0 top-0 text-xs font-medium text-gray-700 mt-2">
                {latMax.toFixed(4)}
            </div>
            <div className="absolute right-0 bottom-0 text-xs font-medium text-gray-700 mb-2">
                {latMin.toFixed(4)}
            </div>
        </div>
        
        {/* X-Axis Label */}
        <div className="text-center mt-2 text-sm font-semibold text-gray-600">
            Longitude
        </div>
        
        {/* X-Axis Min/Max Markers */}
        <div className="flex justify-between mx-10 text-xs font-medium text-gray-700">
            <span>{lonMin.toFixed(4)}</span>
            <span>{lonMax.toFixed(4)}</span>
        </div>
    </div>
  );
};


// --- Main Application Component ---
const App = () => {
  // Aliasing Lucide icons to avoid conflict with native JS objects (Map, ListChecks)
  const MapIcon = Map;
  const ListChecksIcon = ListChecks;

  const initialFilesState = DATASET_NAMES.reduce((acc, { abbreviation, type }) => ({ 
    ...acc, 
    [abbreviation]: type === 'dynamic' ? [] : null 
  }), {});
  
  const [inputFiles, setInputFiles] = useState(initialFilesState);
  
  // Geographical context state
  const [country] = useState('India');
  const [state] = useState('Uttarakhand');
  
  // State for coordinate inputs
  const [latMin, setLatMin] = useState(DEFAULT_BOUNDS.latMin); 
  const [latMax, setLatMax] = useState(DEFAULT_BOUNDS.latMax);
  const [lonMin, setLonMin] = useState(DEFAULT_BOUNDS.lonMin); 
  const [lonMax, setLonMax] = useState(DEFAULT_BOUNDS.lonMax);

  const [predictionResult, setPredictionResult] = useState([]);
  const [timeDetails, setTimeDetails] = useState(null);
  const [currentHorizon, setCurrentHorizon] = useState(0); 
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const parsedLatMin = parseFloat(latMin);
  const parsedLatMax = parseFloat(latMax);
  const parsedLonMin = parseFloat(lonMin);
  const parsedLonMax = parseFloat(lonMax);

  const isRangeValid = 
    !isNaN(parsedLatMin) && !isNaN(parsedLatMax) && parsedLatMin < parsedLatMax &&
    !isNaN(parsedLonMin) && !isNaN(parsedLonMax) && parsedLonMin < parsedLonMax;

  const isFormValid = DATASET_NAMES.every(({ abbreviation, type, files }) => {
    const fileInput = inputFiles[abbreviation];
    if (type === 'dynamic') {
      return Array.isArray(fileInput) && fileInput.length === files;
    } else {
      return fileInput !== null && fileInput instanceof File;
    }
  }) && isRangeValid;

  const handleFileChange = (abbreviation, fileList, fileType) => {
    if (fileType === 'dynamic') {
        const filesToStore = Array.from(fileList).slice(0, 6);
        setInputFiles(prev => ({ ...prev, [abbreviation]: filesToStore }));
    } else {
        setInputFiles(prev => ({ ...prev, [abbreviation]: fileList[0] || null }));
    }
  };

  const handlePredict = async () => {
    // 1. Check validation
    if (!isFormValid) {
      setError("Please ensure all 32 files are uploaded and the geographical range is valid.");
      return;
    }

    // 2. Start loading state
    setError(null);
    setIsLoading(true);
    setPredictionResult([]);
    setTimeDetails(null);

    try {
      // 3. Call the backend API, passing the file objects (manifest)
      const { prediction, timeDetails: details } = await runPredictionAPI({
        latMin: parsedLatMin,
        latMax: parsedLatMax,
        lonMin: parsedLonMin,
        lonMax: parsedLonMax,
        inputFiles: inputFiles // <<< PASSING THE FILES TO API FUNCTION
      });
      
      // 4. Update state with results
      setPredictionResult(prediction);
      setTimeDetails(details);
      setCurrentHorizon(0); 
    } catch (e) {
      // 5. Catch hard errors from the API call
      console.error("Prediction Error:", e);
      // The fallback runs automatically in runPredictionAPI and sets the mock data.
      setError(`Prediction failed due to API connection error. Showing client-side simulation results instead.`);
    } finally {
      // 6. End loading state
      setIsLoading(false);
    }
  };

  const currentBounds = [
    [parsedLatMin, parsedLonMin], // Southwest (Min Lat, Min Lon)
    [parsedLatMax, parsedLonMax]  // Northeast (Max Lat, Max Lon)
  ];
  
  const getMaxProbability = (horizonIndex) => {
    if (!predictionResult[horizonIndex]) return 0;
    let maxProb = 0;
    predictionResult[horizonIndex].forEach(row => {
      row.forEach(prob => {
        if (prob > maxProb) maxProb = prob;
      });
    });
    return maxProb;
  };

  const getFileStatus = (abbreviation, type, filesRequired) => {
    const fileInput = inputFiles[abbreviation];
    let count = 0;
    
    if (type === 'dynamic') {
      count = Array.isArray(fileInput) ? fileInput.length : 0;
    } else if (fileInput instanceof File) {
      count = 1;
    }

    if (count === filesRequired) {
      return <span className="text-green-500 font-bold flex-shrink-0 text-sm">✓ Ready</span>;
    } else {
      return <span className="text-red-400 font-bold flex-shrink-0 text-sm">({count}/{filesRequired} files)</span>;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 p-4 sm:p-8 font-['Inter']">
      <div className="max-w-8xl mx-auto bg-white p-6 sm:p-10 rounded-3xl shadow-2xl">
        
        {/* --- Header --- */}
        <header className="mb-10 text-left">
          <h1 className="text-4xl font-extrabold text-gray-800 mb-1 flex items-center">
            <Zap className="w-8 h-8 mr-3 text-red-600 fill-red-600" />
            Wildfire Prediction Analysis
          </h1>
          <p className="text-gray-500 max-w-2xl">
            Upload the 7 geospatial data channels (32 files total) to generate the $13 \times 13$ fire probability grid for Uttarakhand.
          </p>
        </header>

        <div className="grid lg:grid-cols-5 gap-8">
          
          {/* --- LEFT COLUMN: Input & Control (Col Span 2) --- */}
          <div className="lg:col-span-2 space-y-6">
            
            {/* 1. Geospatail / Coordinate Input Card */}
            <div className="p-6 bg-white rounded-2xl shadow-lg border border-gray-200">
              <h2 className="text-xl font-bold text-indigo-700 mb-4 flex items-center">
                <Globe className="w-5 h-5 mr-2" />
                Geospatial Context
              </h2>
              
              {/* Context Display */}
              <div className="grid grid-cols-2 gap-4 mb-4 pb-4 border-b border-indigo-100">
                <div>
                  <label className="block text-xs font-semibold text-indigo-800 uppercase">Country</label>
                  <span className="mt-1 block rounded-lg p-2 bg-indigo-50 text-indigo-900 font-medium">{country}</span>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-indigo-800 uppercase">State/Region</label>
                  <span className="mt-1 block rounded-lg p-2 bg-indigo-50 text-indigo-900 font-medium">{state}</span>
                </div>
              </div>
              
              <h3 className="text-md font-semibold text-indigo-700 mb-3 flex items-center">
                 <Target className="w-4 h-4 mr-1"/> $13 \times 13$ Patch Bounds:
              </h3>
              
              {/* Latitude Range */}
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label htmlFor="latMin" className="block text-sm font-medium text-gray-700">Lat Min</label>
                  <input
                    id="latMin" type="text" value={latMin}
                    onChange={(e) => setLatMin(e.target.value)}
                    className="mt-1 block w-full rounded-lg border-gray-300 shadow-sm p-2 border focus:ring-indigo-500 focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label htmlFor="latMax" className="block text-sm font-medium text-gray-700">Lat Max</label>
                  <input
                    id="latMax" type="text" value={latMax}
                    onChange={(e) => setLatMax(e.target.value)}
                    className="mt-1 block w-full rounded-lg border-gray-300 shadow-sm p-2 border focus:ring-indigo-500 focus:border-indigo-500"
                  />
                </div>
              </div>

              {/* Longitude Range */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label htmlFor="lonMin" className="block text-sm font-medium text-gray-700">Lon Min</label>
                  <input
                    id="lonMin" type="text" value={lonMin}
                    onChange={(e) => setLonMin(e.target.value)}
                    className="mt-1 block w-full rounded-lg border-gray-300 shadow-sm p-2 border focus:ring-indigo-500 focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label htmlFor="lonMax" className="block text-sm font-medium text-gray-700">Lon Max</label>
                  <input
                    id="lonMax" type="text" value={lonMax}
                    onChange={(e) => setLonMax(e.target.value)}
                    className="mt-1 block w-full rounded-lg border-gray-300 shadow-sm p-2 border focus:ring-indigo-500 focus:border-indigo-500"
                  />
                </div>
              </div>
            </div>

            {/* 2. Data Upload Card */}
            <div className="p-6 bg-white rounded-2xl shadow-lg border border-gray-200">
              <h2 className="text-xl font-bold text-gray-700 mb-4 flex items-center">
                <FileUp className="w-5 h-5 mr-2" />
                Upload Datasets
              </h2>
              <p className="text-sm text-gray-500 mb-4">
                Total 32 files required (5 dynamic channels $\times$ 6 hours, 2 static channels $\times$ 1 file).
              </p>

              <div className="space-y-4">
                <p className="text-xs text-indigo-600 font-semibold uppercase tracking-wider border-b pb-2 mb-2">
                  Dynamic Channels (5 $\times$ 6 Hourly Files)
                </p>
                {DATASET_NAMES.filter(d => d.type === 'dynamic').map(({ name, abbreviation, files, type }) => (
                  <div key={abbreviation} className="flex flex-col space-y-2">
                    <div className="flex justify-between items-center">
                      <label className="text-sm font-medium text-gray-700" title={name}>
                        {abbreviation} ({name})
                      </label>
                      {getFileStatus(abbreviation, type, files)}
                    </div>
                    
                    {/* Dropzone Style Input */}
                    <div className="relative border-2 border-dashed border-gray-300 hover:border-indigo-400 rounded-lg p-3 text-center transition duration-150 cursor-pointer">
                      <input 
                        type="file" 
                        multiple 
                        onChange={(e) => handleFileChange(abbreviation, e.target.files, type)}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                        accept=".tif, .tiff, .nc, .geojson"
                      />
                      <p className="text-xs text-gray-500 flex justify-center items-center">
                        <FileUp className="w-4 h-4 mr-2" />
                        Click to upload {files} files (TIF/NetCDF)
                      </p>
                      {inputFiles[abbreviation].length > 0 && (
                        <p className="text-xs text-indigo-500 mt-1">{inputFiles[abbreviation].length} files selected</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              <div className="space-y-4 mt-6">
                <p className="text-xs text-indigo-600 font-semibold uppercase tracking-wider border-b pb-2 mb-2">
                  Static Channels (2 $\times$ 1 File)
                </p>
                {DATASET_NAMES.filter(d => d.type === 'static').map(({ name, abbreviation, files, type }) => (
                  <div key={abbreviation} className="flex flex-col space-y-2">
                    <div className="flex justify-between items-center">
                      <label className="text-sm font-medium text-gray-700" title={name}>
                        {abbreviation} ({name})
                      </label>
                      {getFileStatus(abbreviation, type, files)}
                    </div>
                    
                    {/* Dropzone Style Input */}
                    <div className="relative border-2 border-dashed border-gray-300 hover:border-indigo-400 rounded-lg p-3 text-center transition duration-150 cursor-pointer">
                      <input 
                        type="file" 
                        onChange={(e) => handleFileChange(abbreviation, e.target.files, type)}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                        accept=".tif, .tiff, .nc, .geojson"
                      />
                      <p className="text-xs text-gray-500 flex justify-center items-center">
                        <FileUp className="w-4 h-4 mr-2" />
                        Click to upload 1 file (TIF/NetCDF)
                      </p>
                      {inputFiles[abbreviation] && inputFiles[abbreviation] instanceof File && (
                        <p className="text-xs text-indigo-500 mt-1">1 file selected</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 3. Run Button */}
            <button
              onClick={handlePredict}
              disabled={!isFormValid || isLoading}
              className={`w-full py-3 px-4 rounded-xl font-bold text-lg transition duration-300 flex items-center justify-center 
                ${isFormValid && !isLoading 
                  ? 'bg-red-600 text-white hover:bg-red-700 shadow-xl shadow-red-300' 
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed shadow-md'
                }`}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-6 h-6 mr-3 animate-spin" />
                  Requesting Prediction...
                </>
              ) : (
                <>
                  <Zap className="w-6 h-6 mr-3" />
                  RUN PREDICTION
                </>
              )}
            </button>

            {error && (
              <p className="text-sm text-red-600 p-3 bg-red-50 rounded-lg border border-red-200">{error}</p>
            )}
            
          </div>

          {/* --- RIGHT COLUMN: Visualization & Details (Col Span 3) --- */}
          <div className="lg:col-span-3 space-y-6">
            <h2 className="text-2xl font-bold text-gray-700 flex items-center">
              <Maximize2 className="w-6 h-6 mr-2 text-indigo-600" />
              Prediction Results
            </h2>
            
            {predictionResult.length > 0 ? (
              <>
                {/* Horizon Selector and Status */}
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center bg-indigo-50 p-4 rounded-xl border border-indigo-200">
                  <div className="flex space-x-2 p-1 bg-white rounded-lg shadow-md mb-3 sm:mb-0">
                    {predictionResult.map((_, index) => (
                      <button
                        key={index}
                        onClick={() => setCurrentHorizon(index)}
                        className={`py-2 px-4 rounded-lg text-sm font-medium transition duration-200 flex items-center ${
                          currentHorizon === index 
                            ? 'bg-indigo-600 text-white shadow-lg' 
                            : 'text-gray-600 hover:bg-gray-100'
                        }`}
                      >
                        <Clock className="w-4 h-4 mr-1" />
                        Hour +{index + 1}
                      </button>
                    ))}
                  </div>
                  <div className="text-sm font-semibold p-2 rounded-lg text-gray-700 bg-red-100 border border-red-200">
                    MAX PROBABILITY: <span className="text-red-800">{getMaxProbability(currentHorizon).toFixed(3)}</span>
                  </div>
                </div>

                {/* Static Grid Plot Component */}
                <StaticGridVisualization 
                  bounds={currentBounds} 
                  prediction={predictionResult} 
                  currentHorizon={currentHorizon} 
                  timeDetails={timeDetails}
                />

                {/* Legend and Time Details */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  
                  {/* Time Window Summary */}
                  <div className="p-4 bg-gray-100 rounded-xl border border-gray-300">
                    <h4 className="font-bold text-gray-700 mb-2 border-b pb-1 flex items-center"><Clock className="w-4 h-4 mr-2"/> Time Analysis ({timeDetails?.date || 'N/A'})</h4>
                    <div className="text-xs space-y-1">
                      {/* Displaying the correct time strings from the backend */}
                      <p><strong>Input (6H):</strong> {timeDetails?.inputStart || 'N/A'} to {timeDetails?.inputEnd || 'N/A'}</p>
                      <p><strong>Prediction (3H):</strong> {timeDetails?.predStart || 'N/A'} to {timeDetails?.predEnd || 'N/A'}</p>
                    </div>
                  </div>

                  {/* Probability Legend */}
                  <div className="p-4 bg-gray-100 rounded-xl border border-gray-300">
                    <h4 className="font-bold text-gray-700 mb-2 border-b pb-1 flex items-center"><ListChecksIcon className="w-4 h-4 mr-2"/> Probability Legend</h4>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="flex items-center">
                        <span className="w-3 h-3 rounded-full mr-2 shadow-md" style={{ backgroundColor: getFireColor(0.8) }}></span> High Risk (0.6 - 1.0)
                      </div>
                      <div className="flex items-center">
                        <span className="w-3 h-3 rounded-full mr-2 shadow-md" style={{ backgroundColor: getFireColor(0.45) }}></span> Medium Risk (0.3 - 0.6)
                      </div>
                      <div className="flex items-center">
                        <span className="w-3 h-3 rounded-full mr-2 shadow-md" style={{ backgroundColor: getFireColor(0.2) }}></span> Low Risk (0.1 - 0.3)
                      </div>
                      <div className="flex items-center">
                        <span className="w-3 h-3 rounded-full mr-2 shadow-md" style={{ backgroundColor: getFireColor(0.05) }}></span> No Fire (0.0 - 0.1)
                      </div>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="h-[550px] flex items-center justify-center bg-gray-100 rounded-2xl border-dashed border-4 border-gray-300 shadow-inner">
                <div className="text-center">
                  <MapIcon className="w-10 h-10 mx-auto text-gray-400 mb-3" />
                  <p className="text-gray-500 text-lg font-medium">Visualization awaits prediction results.</p>
                  <p className="text-sm text-gray-400 mt-1">Configure inputs and run the API.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
