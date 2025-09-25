import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import './AQIGlobe.css';
import TopBar from '../components/real-time-main-page/TopBar';
import Footer from '../components/real-time-main-page/Footer';

// AQI categories and helper functions
const aqiCategories = [
  {
    range: [0, 50],
    level: 'Good',
    color: '#00e400',
    healthImplications:
      'Air quality is satisfactory, and air pollution poses little or no risk.'
  },
  {
    range: [51, 100],
    level: 'Moderate',
    color: '#ffff00',
    healthImplications:
      'Air quality is acceptable, but sensitive individuals may experience minor effects.'
  },
  {
    range: [101, 150],
    level: 'Unhealthy for Sensitive Groups',
    color: '#ff7e00',
    healthImplications:
      'Members of sensitive groups may experience health effects.'
  },
  {
    range: [151, 200],
    level: 'Unhealthy',
    color: '#ff0000',
    healthImplications:
      'Some members of the general public may experience health effects.'
  },
  {
    range: [201, 300],
    level: 'Very Unhealthy',
    color: '#8f3f97',
    healthImplications:
      'Health alert: everyone may experience more serious effects.'
  },
  {
    range: [301, 500],
    level: 'Hazardous',
    color: '#7e0023',
    healthImplications:
      'Emergency conditions: everyone is more likely to be affected.'
  }
];

const mapAQI = (concentration, c_low, c_high, i_low, i_high) => {
  return Math.round(
    ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
  );
};

const calculateAQI = (pollutants) => {
  const pm25AQI = calculatePM25AQI(pollutants.pm2_5 || 0);
  const pm10AQI = calculatePM10AQI(pollutants.pm10 || 0);
  const o3AQI = calculateO3AQI(pollutants.o3 || 0);
  const no2AQI = calculateNO2AQI(pollutants.no2 || 0);
  const so2AQI = calculateSO2AQI(pollutants.so2 || 0);
  const coAQI = calculateCOAQI(pollutants.co || 0);
  return Math.max(pm25AQI, pm10AQI, o3AQI, no2AQI, so2AQI, coAQI);
};

const calculatePM25AQI = (concentration) => {
  if (concentration <= 12) return mapAQI(concentration, 0, 12, 0, 50);
  if (concentration <= 35.4) return mapAQI(concentration, 12.1, 35.4, 51, 100);
  if (concentration <= 55.4) return mapAQI(concentration, 35.5, 55.4, 101, 150);
  if (concentration <= 150.4) return mapAQI(concentration, 55.5, 150.4, 151, 200);
  if (concentration <= 250.4) return mapAQI(concentration, 150.5, 250.4, 201, 300);
  if (concentration <= 500.4) return mapAQI(concentration, 250.5, 500.4, 301, 500);
  return 500;
};

const calculatePM10AQI = (concentration) => {
  if (concentration <= 54) return mapAQI(concentration, 0, 54, 0, 50);
  if (concentration <= 154) return mapAQI(concentration, 55, 154, 51, 100);
  if (concentration <= 254) return mapAQI(concentration, 155, 254, 101, 150);
  if (concentration <= 354) return mapAQI(concentration, 255, 354, 151, 200);
  if (concentration <= 424) return mapAQI(concentration, 355, 424, 201, 300);
  if (concentration <= 604) return mapAQI(concentration, 425, 604, 301, 500);
  return 500;
};

const calculateO3AQI = (concentration) => {
  const ppb = concentration * 0.5;
  if (ppb <= 54) return mapAQI(ppb, 0, 54, 0, 50);
  if (ppb <= 70) return mapAQI(ppb, 55, 70, 51, 100);
  if (ppb <= 85) return mapAQI(ppb, 71, 85, 101, 150);
  if (ppb <= 105) return mapAQI(ppb, 86, 105, 151, 200);
  if (ppb <= 200) return mapAQI(ppb, 106, 200, 201, 300);
  return 500;
};

const calculateNO2AQI = (concentration) => {
  const ppb = concentration * 0.53;
  if (ppb <= 53) return mapAQI(ppb, 0, 53, 0, 50);
  if (ppb <= 100) return mapAQI(ppb, 54, 100, 51, 100);
  if (ppb <= 360) return mapAQI(ppb, 101, 360, 101, 150);
  if (ppb <= 649) return mapAQI(ppb, 361, 649, 151, 200);
  if (ppb <= 1249) return mapAQI(ppb, 650, 1249, 201, 300);
  if (ppb <= 2049) return mapAQI(ppb, 1250, 2049, 301, 500);
  return 500;
};

const calculateSO2AQI = (concentration) => {
  const ppb = concentration * 0.38;
  if (ppb <= 35) return mapAQI(ppb, 0, 35, 0, 50);
  if (ppb <= 75) return mapAQI(ppb, 36, 75, 51, 100);
  if (ppb <= 185) return mapAQI(ppb, 76, 185, 101, 150);
  if (ppb <= 304) return mapAQI(ppb, 186, 304, 151, 200);
  if (ppb <= 604) return mapAQI(ppb, 305, 604, 201, 300);
  if (ppb <= 1004) return mapAQI(ppb, 605, 1004, 301, 500);
  return 500;
};

const calculateCOAQI = (concentration) => {
  const ppm = concentration * 0.000873;
  if (ppm <= 4.4) return mapAQI(ppm, 0, 4.4, 0, 50);
  if (ppm <= 9.4) return mapAQI(ppm, 4.5, 9.4, 51, 100);
  if (ppm <= 12.4) return mapAQI(ppm, 9.5, 12.4, 101, 150);
  if (ppm <= 15.4) return mapAQI(ppm, 12.4, 15.4, 151, 200);
  if (ppm <= 30.4) return mapAQI(ppm, 15.5, 30.4, 201, 300);
  if (ppm <= 50.4) return mapAQI(ppm, 30.5, 50.4, 301, 500);
  return 500;
};

const getAQICategory = (aqi) => {
  for (const category of aqiCategories) {
    if (aqi >= category.range[0] && aqi <= category.range[1]) {
      return category;
    }
  }
  return aqiCategories[aqiCategories.length - 1];
};

// List of around 20 cities for improved performance.
const majorCities = [
  { name: 'New York', country: 'US', lat: 40.7128, lon: -74.0060 },
  { name: 'Los Angeles', country: 'US', lat: 34.0522, lon: -118.2437 },
  { name: 'Chicago', country: 'US', lat: 41.8781, lon: -87.6298 },
  { name: 'Toronto', country: 'CA', lat: 43.6532, lon: -79.3832 },
  { name: 'London', country: 'GB', lat: 51.5074, lon: -0.1278 },
  { name: 'Paris', country: 'FR', lat: 48.8566, lon: 2.3522 },
  { name: 'Berlin', country: 'DE', lat: 52.5200, lon: 13.4050 },
  { name: 'Madrid', country: 'ES', lat: 40.4168, lon: -3.7038 },
  { name: 'Rome', country: 'IT', lat: 41.9028, lon: 12.4964 },
  { name: 'Moscow', country: 'RU', lat: 55.7558, lon: 37.6173 },
  { name: 'Beijing', country: 'CN', lat: 39.9042, lon: 116.4074 },
  { name: 'Tokyo', country: 'JP', lat: 35.6762, lon: 139.6503 },
  { name: 'Seoul', country: 'KR', lat: 37.5665, lon: 126.9780 },
  { name: 'Singapore', country: 'SG', lat: 1.3521, lon: 103.8198 },
  { name: 'Sydney', country: 'AU', lat: -33.8688, lon: 151.2093 },
  { name: 'Dubai', country: 'AE', lat: 25.2048, lon: 55.2708 },
  { name: 'Istanbul', country: 'TR', lat: 41.0082, lon: 28.9784 },
  { name: 'Rio de Janeiro', country: 'BR', lat: -22.9068, lon: -43.1729 },
  { name: 'Buenos Aires', country: 'AR', lat: -34.6037, lon: -58.3816 },
  { name: 'Cape Town', country: 'ZA', lat: -33.9249, lon: 18.4241 }
];

// Simple cache using localStorage
const mockDatabase = (() => {
  const storage = window.localStorage;
  const storeData = (cityName, data) => {
    const storedData = {
      timestamp: new Date().getTime(),
      data
    };
    storage.setItem(`aqi_${cityName}`, JSON.stringify(storedData));
  };
  const getData = (cityName) => {
    const storedData = storage.getItem(`aqi_${cityName}`);
    return storedData ? JSON.parse(storedData) : null;
  };
  const isDataFresh = (cityName) => {
    const storedData = getData(cityName);
    if (!storedData) return false;
    const oneHour = 60 * 60 * 1000;
    return new Date().getTime() - storedData.timestamp < oneHour;
  };
  return { storeData, getData, isDataFresh };
})();

// Helper: Convert lat/lon to 3D coordinates on the sphere
const latLonToVector3 = (lat, lon, radius) => {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  const x = -radius * Math.sin(phi) * Math.cos(theta);
  const y = radius * Math.cos(phi);
  const z = radius * Math.sin(phi) * Math.sin(theta);
  return new THREE.Vector3(x, y, z);
};
const AQIGlobe = () => {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const globeRef = useRef(null);
  const markersRef = useRef([]);
  const raycasterRef = useRef(new THREE.Raycaster());
  const mouseRef = useRef(new THREE.Vector2());
  const tooltipRef = useRef(null);
  const popupRef = useRef(null);

  const [cityData, setCityData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [darkMode, setDarkMode] = useState(true);
  const [selectedCity, setSelectedCity] = useState(null);
  const [isGlobeRotating, setIsGlobeRotating] = useState(true);

  const apiKey = 'aac405e628f9c30a047d3de13192a7f7'; // Replace with your API key

  const toggleDarkMode = () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    localStorage.setItem('darkMode', newMode);
    document.body.className = newMode ? 'dark-mode' : 'light-mode';
    if (globeRef.current) {
      globeRef.current.material.map = createGlobeTexture();
      globeRef.current.material.needsUpdate = true;
    }
    if (sceneRef.current) {
      sceneRef.current.background = new THREE.Color(newMode ? 0x000000 : 0xf0f0f0);
    }
    
    // Refresh markers on theme change
    refreshMarkers();
  };

  const toggleGlobeRotation = () => {
    setIsGlobeRotating(!isGlobeRotating);
  };

  const createGlobeTexture = (isDark=true) => {
    const loader = new THREE.TextureLoader();
    return loader.load(
      isDark
        ? 'https://cdn.jsdelivr.net/npm/three-globe/example/img/earth-night.jpg'
        : 'https://cdn.jsdelivr.net/npm/three-globe/example/img/earth-blue-marble.jpg'
    );
  };

  // Initialize scene once (or when darkMode changes)
  useEffect(() => {
    if (!containerRef.current) return;

    // Clean container first (in case of reinitialization)
    containerRef.current.innerHTML = '';

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(darkMode ? 0x000000 : 0xf0f0f0);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      60,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.z = 200;
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(
      containerRef.current.clientWidth,
      containerRef.current.clientHeight
    );
    renderer.setPixelRatio(window.devicePixelRatio);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.rotateSpeed = 0.5;
    controls.minDistance = 120;
    controls.maxDistance = 300;
    controlsRef.current = controls;

    // Create the globe
    const globeRadius = 100;
    const globeGeometry = new THREE.SphereGeometry(globeRadius, 64, 64);
    const globeMaterial = new THREE.MeshPhongMaterial({
      map: createGlobeTexture(darkMode),
      bumpMap: new THREE.TextureLoader().load(
        'https://cdn.jsdelivr.net/npm/three-globe/example/img/earth-topology.png'
      ),
      bumpScale: 0.5,
      specular: new THREE.Color(0x333333),
      shininess: 5
    });
    const globe = new THREE.Mesh(globeGeometry, globeMaterial);
    scene.add(globe);
    globeRef.current = globe;

    scene.add(new THREE.AmbientLight(0x404040, 1.5));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    // Create tooltip element
    const tooltip = document.createElement('div');
    tooltip.className = 'city-tooltip';
    tooltip.style.display = 'none';
    containerRef.current.appendChild(tooltip);
    tooltipRef.current = tooltip;

    // Create popup element
    const popup = document.createElement('div');
    popup.className = 'city-popup';
    popup.style.display = 'none';
    containerRef.current.appendChild(popup);
    popupRef.current = popup;

    // Resize handler
    const handleResize = () => {
      if (!containerRef.current) return;
      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);

    // Debounced mouse move for tooltip
    let mouseMoveTimeout = null;
    const handleMouseMove = (event) => {
      if (mouseMoveTimeout) clearTimeout(mouseMoveTimeout);
      mouseMoveTimeout = setTimeout(() => {
        if (!containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
        mouseRef.current.x =
          ((event.clientX - rect.left) / containerRef.current.clientWidth) * 2 - 1;
        mouseRef.current.y =
          -((event.clientY - rect.top) / containerRef.current.clientHeight) * 2 + 1;
        tooltip.style.left = `${event.clientX + 15}px`;
        tooltip.style.top = `${event.clientY + 15}px`;
      }, 20);
    };
    containerRef.current.addEventListener('mousemove', handleMouseMove);

    // Click handler to show popup
    const handleClick = (event) => {
      raycasterRef.current.setFromCamera(mouseRef.current, camera);
      const intersects = raycasterRef.current.intersectObjects(markersRef.current);
      if (intersects.length > 0) {
        const city = intersects[0].object.userData;
        setSelectedCity(city);
        updatePopup(city, event.clientX, event.clientY);
      } else {
        setSelectedCity(null);
        if (popupRef.current) popupRef.current.style.display = 'none';
      }
    };
    containerRef.current.addEventListener('click', handleClick);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      // Show tooltip if no popup is open
      if (markersRef.current.length > 0 && !selectedCity) {
        raycasterRef.current.setFromCamera(mouseRef.current, camera);
        const intersects = raycasterRef.current.intersectObjects(markersRef.current);
        if (intersects.length > 0) {
          showTooltip(intersects[0].object.userData);
        } else {
          hideTooltip();
        }
      }
      
      // Rotate globe only if rotation is enabled
      if (globeRef.current && isGlobeRotating && !controls.isDragging) {
        globeRef.current.rotation.y += 0.0005;
        
        // Rotate markers with the globe
        markersRef.current.forEach(marker => {
          // Calculate marker's original position
          const position = marker.userData.position;
          if (position) {
            // Apply the same rotation to the marker
            const rotatedPosition = position.clone();
            rotatedPosition.applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.0005);
            marker.position.copy(rotatedPosition);
          }
        });
      }
      
      renderer.render(scene, camera);
    };
    animate();

    // Cleanup on unmount
    return () => {
      if (containerRef.current) {
        containerRef.current.removeEventListener('mousemove', handleMouseMove);
        containerRef.current.removeEventListener('click', handleClick);
      }
      window.removeEventListener('resize', handleResize);
      if (
        rendererRef.current &&
        containerRef.current &&
        containerRef.current.contains(rendererRef.current.domElement)
      ) {
        containerRef.current.removeChild(rendererRef.current.domElement);
      }
      if (
        tooltipRef.current &&
        containerRef.current &&
        containerRef.current.contains(tooltipRef.current)
      ) {
        containerRef.current.removeChild(tooltipRef.current);
      }
      if (
        popupRef.current &&
        containerRef.current &&
        containerRef.current.contains(popupRef.current)
      ) {
        containerRef.current.removeChild(popupRef.current);
      }
    };
  }, [darkMode]); // Initialize only once (or on darkMode change)

  // Fetch air quality data on mount and every hour
  useEffect(() => {
    fetchAllCityData();
    const interval = setInterval(fetchAllCityData, 60 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  // Update markers whenever cityData updates.
  useEffect(() => {
    if (!sceneRef.current || cityData.length === 0) return;
    refreshMarkers();
  }, [cityData]);

  const refreshMarkers = () => {
    if (!sceneRef.current) return;
    // Remove existing markers
    markersRef.current.forEach((marker) => sceneRef.current.remove(marker));
    markersRef.current = [];
    // Add new markers
    cityData.forEach((city) => addCityMarker(city));
  };

  const fetchAllCityData = async () => {
    setLoading(true);
    const dataArr = [];
    const batchSize = 5;
    for (let i = 0; i < majorCities.length; i += batchSize) {
      const batch = majorCities.slice(i, i + batchSize);
      await Promise.all(batch.map((city) => fetchCityAQIData(city, dataArr)));
      if (i + batchSize < majorCities.length)
        await new Promise((r) => setTimeout(r, 1000));
    }
    setCityData(dataArr);
    setLastUpdated(new Date());
    setLoading(false);
  };

  const fetchCityAQIData = async (city, dataArr) => {
    if (mockDatabase.isDataFresh(city.name)) {
      const stored = mockDatabase.getData(city.name);
      dataArr.push({ ...city, ...stored.data, fromCache: true });
      return;
    }
    try {
      const response = await fetch(
        `https://api.openweathermap.org/data/2.5/air_pollution?lat=${city.lat}&lon=${city.lon}&appid=${apiKey}`
      );
      if (!response.ok)
        throw new Error(`Failed to fetch data for ${city.name}`);
      const result = await response.json();
      if (result?.list?.length > 0) {
        const pollutants = result.list[0].components;
        const aqi = calculateAQI(pollutants);
        const aqiCategory = getAQICategory(aqi);
        const cityAQIData = {
          ...city,
          pollutants,
          aqi,
          aqiCategory,
          fromCache: false
        };
        mockDatabase.storeData(city.name, { pollutants, aqi, aqiCategory });
        dataArr.push(cityAQIData);
      }
    } catch (error) {
      console.error(`Error fetching data for ${city.name}:`, error);
      const stored = mockDatabase.getData(city.name);
      if (stored) {
        dataArr.push({
          ...city,
          ...stored.data,
          fromCache: true,
          dataStale: true
        });
      }
    }
  };

  // Create a visible marker (pin-style marker)
  const addCityMarker = (city) => {
    if (!sceneRef.current || !city || !city.aqiCategory) return;
    const globeRadius = 100;
    const position = latLonToVector3(city.lat, city.lon, globeRadius + 1);
    
    // Create a pin-like marker with SVG
    const markerCanvas = document.createElement('canvas');
    markerCanvas.width = 64;
    markerCanvas.height = 64;
    const context = markerCanvas.getContext('2d');
    
    // Calculate marker size based on PM2.5 concentration (with min and max limits)
    const pm25 = city.pollutants.pm2_5 || 0;
    const minSize = 8; // Minimum pin size
    const maxSize = 16; // Maximum pin size
    const pinSize = Math.min(maxSize, Math.max(minSize, 8 + (pm25 / 20)));

    // Draw a pin-like marker
    context.beginPath();
    // Pin head (circle)
    context.arc(32, 24, pinSize, 0, 2 * Math.PI);
    context.fillStyle = city.aqiCategory.color;
    context.fill();
    context.lineWidth = 2;
    context.strokeStyle = '#ffffff';
    context.stroke();
    
    // Pin tail (triangle)
    context.beginPath();
    context.moveTo(32 - pinSize/2, 24 + pinSize/2);
    context.lineTo(32, 32 + pinSize);
    context.lineTo(32 + pinSize/2, 24 + pinSize/2);
    context.closePath();
    context.fillStyle = city.aqiCategory.color;
    context.fill();
    context.lineWidth = 2;
    context.strokeStyle = '#ffffff';
    context.stroke();

    const texture = new THREE.CanvasTexture(markerCanvas);
    const spriteMaterial = new THREE.SpriteMaterial({ 
      map: texture,
      transparent: true,
      depthTest: false
    });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.scale.set(10, 10, 1);
    sprite.position.copy(position);
    sprite.userData = {
      ...city,
      position // Store the original position for rotation tracking
    };
    sceneRef.current.add(sprite);
    markersRef.current.push(sprite);
  };

  const showTooltip = (city) => {
    if (!tooltipRef.current) return;
    tooltipRef.current.innerHTML = `<strong>${city.name}, ${city.country}</strong><br>
      <span style="color: ${city.aqiCategory.color}">AQI: ${city.aqi} - ${city.aqiCategory.level}</span>`;
    tooltipRef.current.style.display = 'block';
  };

  const hideTooltip = () => {
    if (tooltipRef.current) tooltipRef.current.style.display = 'none';
  };

  const updatePopup = (city, x, y) => {
    if (!popupRef.current) return;
    popupRef.current.innerHTML = `
      <div class="popup-header" style="background-color: ${city.aqiCategory.color}">
          <h3>${city.name}, ${city.country}</h3>
          <div class="aqi-value">AQI: ${city.aqi} - ${city.aqiCategory.level}</div>
      </div>
      <div class="popup-content">
          <p>${city.aqiCategory.healthImplications}</p>
          <h4>Pollutant Details</h4>
          <ul>
            <li>PM2.5: ${city.pollutants.pm2_5 || 'N/A'} μg/m³</li>
            <li>PM10: ${city.pollutants.pm10 || 'N/A'} μg/m³</li>
            <li>O3: ${city.pollutants.o3 || 'N/A'} μg/m³</li>
            <li>NO2: ${city.pollutants.no2 || 'N/A'} μg/m³</li>
            <li>SO2: ${city.pollutants.so2 || 'N/A'} μg/m³</li>
            <li>CO: ${city.pollutants.co || 'N/A'} μg/m³</li>
          </ul>
          <div class="popup-footer">
              ${
                city.fromCache
                  ? city.dataStale
                    ? '<span class="cache-notice stale">Data may be outdated</span>'
                    : '<span class="cache-notice">From cache</span>'
                  : '<span class="fresh-data">Latest data</span>'
              }
              <button class="close-popup">Close</button>
          </div>
      </div>
    `;
    const maxWidth = containerRef.current
      ? containerRef.current.clientWidth - 320
      : 0;
    const maxHeight = containerRef.current
      ? containerRef.current.clientHeight - 400
      : 0;
    popupRef.current.style.left = `${Math.min(x, maxWidth)}px`;
    popupRef.current.style.top = `${Math.min(y, maxHeight)}px`;
    popupRef.current.style.display = 'block';

    const closeBtn = popupRef.current.querySelector('.close-popup');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => {
        if (popupRef.current) popupRef.current.style.display = 'none';
        setSelectedCity(null);
      });
    }
  };

  // Generate the AQI gradient CSS for the linear gradient
  const generateAQIGradient = () => {
    let gradientStops = '';
    aqiCategories.forEach((cat, index) => {
      const percentage = (index / (aqiCategories.length - 1)) * 100;
      gradientStops += `${cat.color} ${percentage}%`;
      if (index < aqiCategories.length - 1) gradientStops += ', ';
    });
    return `linear-gradient(to right, ${gradientStops})`;
  };

  return (
    <div className={`aqi-globe-page ${darkMode ? 'dark-mode' : 'light-mode'}`}>
      <TopBar darkMode={darkMode} toggleDarkMode={toggleDarkMode} />
      <div className="globe-container">
        <div className="globe-visualization" ref={containerRef}></div>
        <div className="globe-info-panel">
          <h2>Global Air Quality Index</h2>
          <p>Real-time air quality data for major cities worldwide</p>
          {loading ? (
            <div className="loading-indicator">
              <div className="spinner"></div>
              <p>Loading air quality data...</p>
            </div>
          ) : (
            <div className="globe-stats">
              <div className="stat-item">
                <span className="stat-label">Cities Monitored:</span>
                <span className="stat-value">{cityData.length}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Last Updated:</span>
                <span className="stat-value">
                  {lastUpdated
                    ? new Intl.DateTimeFormat('en-US', {
                        hour: '2-digit',
                        minute: '2-digit',
                        day: '2-digit',
                        month: 'short'
                      }).format(lastUpdated)
                    : 'N/A'}
                </span>
              </div>
              <div className="control-buttons">
                <button
                  className="refresh-button"
                  onClick={fetchAllCityData}
                  disabled={loading}
                >
                  Refresh Data
                </button>
                <button
                  className={`rotation-button ${isGlobeRotating ? 'active' : ''}`}
                  onClick={toggleGlobeRotation}
                >
                  {isGlobeRotating ? 'Stop Rotation' : 'Start Rotation'}
                </button>
              </div>
            </div>
          )}
          <div className="aqi-legend">
            <h4>AQI Legend</h4>
            <div className="legend-bar">
              <div 
                className="gradient-bar" 
                style={{ background: generateAQIGradient() }}
              ></div>
              <div className="scale-markers">
                {aqiCategories.map((cat, i) => (
                  <div key={i} className="scale-marker">
                    <span className="scale-value">{cat.range[0]}</span>
                    <span className="scale-label">{cat.level}</span>
                  </div>
                ))}
                <div className="scale-marker">
                  <span className="scale-value">{aqiCategories[aqiCategories.length-1].range[1]}</span>
                </div>
              </div>
            </div>
          </div>
          <div className="marker-legend">
            <h4>Marker Guide</h4>
            <div className="marker-explanation">
              <div className="marker-example">
                <svg width="32" height="32" viewBox="0 0 32 32">
                  <circle cx="16" cy="12" r="8" fill="#00e400" stroke="#ffffff" strokeWidth="1.5" />
                  <path d="M12,12 L16,24 L20,12 Z" fill="#00e400" stroke="#ffffff" strokeWidth="1.5" />
                </svg>
              </div>
              <div className="marker-text">
                <p>Each marker represents a city's AQI data</p>
                <p><strong>Color:</strong> Indicates AQI category</p>
                <p><strong>Size:</strong> Indicates PM2.5 concentration</p>
              </div>
            </div>
          </div>
          <div className="globe-instructions">
            <h4>Instructions</h4>
            <ul>
              <li>Drag to rotate the globe</li>
              <li>Scroll to zoom in/out</li>
              <li>Hover over a pin for quick details</li>
              <li>Click a pin for comprehensive information</li>
              <li>Toggle rotation with the button above</li>
            </ul>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default AQIGlobe;