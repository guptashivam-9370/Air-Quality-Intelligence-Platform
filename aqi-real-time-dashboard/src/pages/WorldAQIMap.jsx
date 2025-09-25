import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, ZoomControl } from 'react-leaflet';
import { useLocation } from 'react-router-dom';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './WorldAQIMap.css';
import TopBar from '../components/real-time-main-page/TopBar';
import Footer from '../components/real-time-main-page/Footer';

// Fix for marker icons in React-Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// AQI categories and helper functions
const aqiCategories = [
    {
        range: [0, 50],
        level: 'Good',
        color: '#00e400',
        healthImplications:
            'Air quality is satisfactory, and air pollution poses little or no risk.',
    },
    {
        range: [51, 100],
        level: 'Moderate',
        color: '#ffff00',
        healthImplications:
            'Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.',
    },
    {
        range: [101, 150],
        level: 'Unhealthy for Sensitive Groups',
        color: '#ff7e00',
        healthImplications:
            'Members of sensitive groups may experience health effects. The general public is less likely to be affected.',
    },
    {
        range: [151, 200],
        level: 'Unhealthy',
        color: '#ff0000',
        healthImplications:
            'Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.',
    },
    {
        range: [201, 300],
        level: 'Very Unhealthy',
        color: '#8f3f97',
        healthImplications:
            'Health alert: The risk of health effects is increased for everyone.',
    },
    {
        range: [301, 500],
        level: 'Hazardous',
        color: '#7e0023',
        healthImplications:
            'Health warning of emergency conditions: everyone is more likely to be affected.',
    },
];

const mapAQI = (concentration, c_low, c_high, i_low, i_high) => {
    return Math.round(
        ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
    );
};

const calculateAQI = (pollutants) => {
    const pm25 = pollutants.pm2_5 || 0;
    const pm10 = pollutants.pm10 || 0;
    const o3 = pollutants.o3 || 0;
    const no2 = pollutants.no2 || 0;
    const so2 = pollutants.so2 || 0;
    const co = pollutants.co || 0;

    const pm25AQI = calculatePM25AQI(pm25);
    const pm10AQI = calculatePM10AQI(pm10);
    const o3AQI = calculateO3AQI(o3);
    const no2AQI = calculateNO2AQI(no2);
    const so2AQI = calculateSO2AQI(so2);
    const coAQI = calculateCOAQI(co);

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

// List of major cities with their coordinates
// List of major cities with their coordinates
const majorCities = [
    // Original cities
    { name: 'New York', country: 'US', lat: 40.7128, lon: -74.0060 },
    { name: 'London', country: 'GB', lat: 51.5074, lon: -0.1278 },
    { name: 'Paris', country: 'FR', lat: 48.8566, lon: 2.3522 },
    { name: 'Tokyo', country: 'JP', lat: 35.6762, lon: 139.6503 },
    { name: 'Beijing', country: 'CN', lat: 39.9042, lon: 116.4074 },
    { name: 'Delhi', country: 'IN', lat: 28.6139, lon: 77.2090 },
    { name: 'Mumbai', country: 'IN', lat: 19.0760, lon: 72.8777 },
    { name: 'Sydney', country: 'AU', lat: -33.8688, lon: 151.2093 },
    { name: 'Rio de Janeiro', country: 'BR', lat: -22.9068, lon: -43.1729 },
    { name: 'Cairo', country: 'EG', lat: 30.0444, lon: 31.2357 },
    { name: 'Lagos', country: 'NG', lat: 6.5244, lon: 3.3792 },
    { name: 'Moscow', country: 'RU', lat: 55.7558, lon: 37.6173 },
    { name: 'Los Angeles', country: 'US', lat: 34.0522, lon: -118.2437 },
    { name: 'Berlin', country: 'DE', lat: 52.5200, lon: 13.4050 },
    { name: 'Mexico City', country: 'MX', lat: 19.4326, lon: -99.1332 },
    { name: 'Singapore', country: 'SG', lat: 1.3521, lon: 103.8198 },
    { name: 'Bangkok', country: 'TH', lat: 13.7563, lon: 100.5018 },
    { name: 'Toronto', country: 'CA', lat: 43.6532, lon: -79.3832 },
    { name: 'Buenos Aires', country: 'AR', lat: -34.6037, lon: -58.3816 },
    { name: 'Istanbul', country: 'TR', lat: 41.0082, lon: 28.9784 },
    { name: 'Chicago', country: 'US', lat: 41.8781, lon: -87.6298 },
    { name: 'Shanghai', country: 'CN', lat: 31.2304, lon: 121.4737 },
    { name: 'Seoul', country: 'KR', lat: 37.5665, lon: 126.9780 },
    { name: 'Hong Kong', country: 'HK', lat: 22.3193, lon: 114.1694 },
    { name: 'Barcelona', country: 'ES', lat: 41.3851, lon: 2.1734 },
    { name: 'Madrid', country: 'ES', lat: 40.4168, lon: -3.7038 },
    { name: 'Rome', country: 'IT', lat: 41.9028, lon: 12.4964 },
    { name: 'Dubai', country: 'AE', lat: 25.2048, lon: 55.2708 },
    { name: 'Johannesburg', country: 'ZA', lat: -26.2041, lon: 28.0473 },
    { name: 'Cape Town', country: 'ZA', lat: -33.9249, lon: 18.4241 },
    
    // Additional major cities (150 more)
    { name: 'San Francisco', country: 'US', lat: 37.7749, lon: -122.4194 },
    { name: 'Washington DC', country: 'US', lat: 38.9072, lon: -77.0369 },
    { name: 'Boston', country: 'US', lat: 42.3601, lon: -71.0589 },
    { name: 'Seattle', country: 'US', lat: 47.6062, lon: -122.3321 },
    { name: 'Miami', country: 'US', lat: 25.7617, lon: -80.1918 },
    { name: 'Houston', country: 'US', lat: 29.7604, lon: -95.3698 },
    { name: 'Dallas', country: 'US', lat: 32.7767, lon: -96.7970 },
    { name: 'Philadelphia', country: 'US', lat: 39.9526, lon: -75.1652 },
    { name: 'Atlanta', country: 'US', lat: 33.7490, lon: -84.3880 },
    { name: 'Denver', country: 'US', lat: 39.7392, lon: -104.9903 },
    { name: 'Phoenix', country: 'US', lat: 33.4484, lon: -112.0740 },
    { name: 'San Diego', country: 'US', lat: 32.7157, lon: -117.1611 },
    { name: 'Las Vegas', country: 'US', lat: 36.1699, lon: -115.1398 },
    { name: 'Vancouver', country: 'CA', lat: 49.2827, lon: -123.1207 },
    { name: 'Montreal', country: 'CA', lat: 45.5017, lon: -73.5673 },
    { name: 'Ottawa', country: 'CA', lat: 45.4215, lon: -75.6972 },
    { name: 'Calgary', country: 'CA', lat: 51.0447, lon: -114.0719 },
    { name: 'Edmonton', country: 'CA', lat: 53.5461, lon: -113.4938 },
    { name: 'Manchester', country: 'GB', lat: 53.4808, lon: -2.2426 },
    { name: 'Birmingham', country: 'GB', lat: 52.4862, lon: -1.8904 },
    { name: 'Glasgow', country: 'GB', lat: 55.8642, lon: -4.2518 },
    { name: 'Liverpool', country: 'GB', lat: 53.4084, lon: -2.9916 },
    { name: 'Edinburgh', country: 'GB', lat: 55.9533, lon: -3.1883 },
    { name: 'Dublin', country: 'IE', lat: 53.3498, lon: -6.2603 },
    { name: 'Belfast', country: 'GB', lat: 54.5973, lon: -5.9301 },
    { name: 'Amsterdam', country: 'NL', lat: 52.3676, lon: 4.9041 },
    { name: 'Rotterdam', country: 'NL', lat: 51.9244, lon: 4.4777 },
    { name: 'The Hague', country: 'NL', lat: 52.0705, lon: 4.3007 },
    { name: 'Brussels', country: 'BE', lat: 50.8503, lon: 4.3517 },
    { name: 'Antwerp', country: 'BE', lat: 51.2194, lon: 4.4025 },
    { name: 'Lyon', country: 'FR', lat: 45.7640, lon: 4.8357 },
    { name: 'Marseille', country: 'FR', lat: 43.2965, lon: 5.3698 },
    { name: 'Nice', country: 'FR', lat: 43.7102, lon: 7.2620 },
    { name: 'Toulouse', country: 'FR', lat: 43.6047, lon: 1.4442 },
    { name: 'Bordeaux', country: 'FR', lat: 44.8378, lon: -0.5792 },
    { name: 'Hamburg', country: 'DE', lat: 53.5511, lon: 9.9937 },
    { name: 'Munich', country: 'DE', lat: 48.1351, lon: 11.5820 },
    { name: 'Frankfurt', country: 'DE', lat: 50.1109, lon: 8.6821 },
    { name: 'Cologne', country: 'DE', lat: 50.9375, lon: 6.9603 },
    { name: 'Stuttgart', country: 'DE', lat: 48.7758, lon: 9.1829 },
    { name: 'Vienna', country: 'AT', lat: 48.2082, lon: 16.3738 },
    { name: 'Salzburg', country: 'AT', lat: 47.8095, lon: 13.0550 },
    { name: 'Zurich', country: 'CH', lat: 47.3769, lon: 8.5417 },
    { name: 'Geneva', country: 'CH', lat: 46.2044, lon: 6.1432 },
    { name: 'Basel', country: 'CH', lat: 47.5596, lon: 7.5886 },
    { name: 'Milan', country: 'IT', lat: 45.4642, lon: 9.1900 },
    { name: 'Naples', country: 'IT', lat: 40.8518, lon: 14.2681 },
    { name: 'Turin', country: 'IT', lat: 45.0703, lon: 7.6869 },
    { name: 'Florence', country: 'IT', lat: 43.7696, lon: 11.2558 },
    { name: 'Venice', country: 'IT', lat: 45.4408, lon: 12.3155 },
    { name: 'Seville', country: 'ES', lat: 37.3891, lon: -5.9845 },
    { name: 'Valencia', country: 'ES', lat: 39.4699, lon: -0.3763 },
    { name: 'Malaga', country: 'ES', lat: 36.7213, lon: -4.4213 },
    { name: 'Lisbon', country: 'PT', lat: 38.7223, lon: -9.1393 },
    { name: 'Porto', country: 'PT', lat: 41.1579, lon: -8.6291 },
    { name: 'Athens', country: 'GR', lat: 37.9838, lon: 23.7275 },
    { name: 'Thessaloniki', country: 'GR', lat: 40.6401, lon: 22.9444 },
    { name: 'Stockholm', country: 'SE', lat: 59.3293, lon: 18.0686 },
    { name: 'Gothenburg', country: 'SE', lat: 57.7089, lon: 11.9746 },
    { name: 'Malmo', country: 'SE', lat: 55.6050, lon: 13.0038 },
    { name: 'Oslo', country: 'NO', lat: 59.9139, lon: 10.7522 },
    { name: 'Copenhagen', country: 'DK', lat: 55.6761, lon: 12.5683 },
    { name: 'Helsinki', country: 'FI', lat: 60.1699, lon: 24.9384 },
    { name: 'Reykjavik', country: 'IS', lat: 64.1466, lon: -21.9426 },
    { name: 'Warsaw', country: 'PL', lat: 52.2297, lon: 21.0122 },
    { name: 'Krakow', country: 'PL', lat: 50.0647, lon: 19.9450 },
    { name: 'Wroclaw', country: 'PL', lat: 51.1079, lon: 17.0385 },
    { name: 'Prague', country: 'CZ', lat: 50.0755, lon: 14.4378 },
    { name: 'Brno', country: 'CZ', lat: 49.1951, lon: 16.6068 },
    { name: 'Budapest', country: 'HU', lat: 47.4979, lon: 19.0402 },
    { name: 'Bratislava', country: 'SK', lat: 48.1486, lon: 17.1077 },
    { name: 'Ljubljana', country: 'SI', lat: 46.0569, lon: 14.5058 },
    { name: 'Zagreb', country: 'HR', lat: 45.8150, lon: 15.9819 },
    { name: 'Belgrade', country: 'RS', lat: 44.7866, lon: 20.4489 },
    { name: 'Sarajevo', country: 'BA', lat: 43.8563, lon: 18.4131 },
    { name: 'Skopje', country: 'MK', lat: 41.9973, lon: 21.4280 },
    { name: 'Tirana', country: 'AL', lat: 41.3275, lon: 19.8187 },
    { name: 'Sofia', country: 'BG', lat: 42.6977, lon: 23.3219 },
    { name: 'Bucharest', country: 'RO', lat: 44.4268, lon: 26.1025 },
    { name: 'Kiev', country: 'UA', lat: 50.4501, lon: 30.5234 },
    { name: 'Odessa', country: 'UA', lat: 46.4825, lon: 30.7233 },
    { name: 'Minsk', country: 'BY', lat: 53.9045, lon: 27.5615 },
    { name: 'Riga', country: 'LV', lat: 56.9496, lon: 24.1052 },
    { name: 'Tallinn', country: 'EE', lat: 59.4370, lon: 24.7536 },
    { name: 'Vilnius', country: 'LT', lat: 54.6872, lon: 25.2797 },
    { name: 'St. Petersburg', country: 'RU', lat: 59.9343, lon: 30.3351 },
    { name: 'Novosibirsk', country: 'RU', lat: 55.0084, lon: 82.9357 },
    { name: 'Yekaterinburg', country: 'RU', lat: 56.8389, lon: 60.6057 },
    { name: 'Kazan', country: 'RU', lat: 55.8304, lon: 49.0661 },
    { name: 'Vladivostok', country: 'RU', lat: 43.1332, lon: 131.9113 },
    { name: 'Ankara', country: 'TR', lat: 39.9334, lon: 32.8597 },
    { name: 'Izmir', country: 'TR', lat: 38.4237, lon: 27.1428 },
    { name: 'Antalya', country: 'TR', lat: 36.8969, lon: 30.7133 },
    { name: 'Baku', country: 'AZ', lat: 40.4093, lon: 49.8671 },
    { name: 'Tbilisi', country: 'GE', lat: 41.7151, lon: 44.8271 },
    { name: 'Yerevan', country: 'AM', lat: 40.1792, lon: 44.4991 },
    { name: 'Tehran', country: 'IR', lat: 35.6892, lon: 51.3890 },
    { name: 'Isfahan', country: 'IR', lat: 32.6546, lon: 51.6680 },
    { name: 'Baghdad', country: 'IQ', lat: 33.3152, lon: 44.3661 },
    { name: 'Kuwait City', country: 'KW', lat: 29.3759, lon: 47.9774 },
    { name: 'Riyadh', country: 'SA', lat: 24.7136, lon: 46.6753 },
    { name: 'Jeddah', country: 'SA', lat: 21.4858, lon: 39.1925 },
    { name: 'Muscat', country: 'OM', lat: 23.5880, lon: 58.3829 },
    { name: 'Doha', country: 'QA', lat: 25.2854, lon: 51.5310 },
    { name: 'Abu Dhabi', country: 'AE', lat: 24.4539, lon: 54.3773 },
    { name: 'Manama', country: 'BH', lat: 26.2285, lon: 50.5860 },
    { name: 'Amman', country: 'JO', lat: 31.9454, lon: 35.9284 },
    { name: 'Damascus', country: 'SY', lat: 33.5138, lon: 36.2765 },
    { name: 'Beirut', country: 'LB', lat: 33.8938, lon: 35.5018 },
    { name: 'Jerusalem', country: 'IL', lat: 31.7683, lon: 35.2137 },
    { name: 'Tel Aviv', country: 'IL', lat: 32.0853, lon: 34.7818 },
    { name: 'Alexandria', country: 'EG', lat: 31.2001, lon: 29.9187 },
    { name: 'Casablanca', country: 'MA', lat: 33.5731, lon: -7.5898 },
    { name: 'Rabat', country: 'MA', lat: 34.0209, lon: -6.8416 },
    { name: 'Tunis', country: 'TN', lat: 36.8065, lon: 10.1815 },
    { name: 'Tripoli', country: 'LY', lat: 32.8872, lon: 13.1913 },
    { name: 'Algiers', country: 'DZ', lat: 36.7538, lon: 3.0588 },
    { name: 'Khartoum', country: 'SD', lat: 15.5007, lon: 32.5599 },
    { name: 'Addis Ababa', country: 'ET', lat: 9.0320, lon: 38.7412 },
    { name: 'Nairobi', country: 'KE', lat: -1.2921, lon: 36.8219 },
    { name: 'Dar es Salaam', country: 'TZ', lat: -6.7924, lon: 39.2083 },
    { name: 'Kampala', country: 'UG', lat: 0.3476, lon: 32.5825 },
    { name: 'Kigali', country: 'RW', lat: -1.9706, lon: 30.1044 },
    { name: 'Lusaka', country: 'ZM', lat: -15.3875, lon: 28.3228 },
    { name: 'Harare', country: 'ZW', lat: -17.8292, lon: 31.0522 },
    { name: 'Windhoek', country: 'NA', lat: -22.5609, lon: 17.0658 },
    { name: 'Gaborone', country: 'BW', lat: -24.6282, lon: 25.9231 },
    { name: 'Durban', country: 'ZA', lat: -29.8587, lon: 31.0218 },
    { name: 'Port Louis', country: 'MU', lat: -20.1609, lon: 57.5012 },
    { name: 'Antananarivo', country: 'MG', lat: -18.8792, lon: 47.5079 },
    { name: 'Abidjan', country: 'CI', lat: 5.3600, lon: -4.0083 },
    { name: 'Dakar', country: 'SN', lat: 14.7167, lon: -17.4677 },
    { name: 'Accra', country: 'GH', lat: 5.6037, lon: -0.1870 },
    { name: 'Lome', country: 'TG', lat: 6.1319, lon: 1.2228 },
    { name: 'Porto-Novo', country: 'BJ', lat: 6.4969, lon: 2.6283 },
    { name: 'Abuja', country: 'NG', lat: 9.0765, lon: 7.3986 },
    { name: 'Bamako', country: 'ML', lat: 12.6392, lon: -8.0029 },
    { name: 'Conakry', country: 'GN', lat: 9.6412, lon: -13.5784 },
    { name: 'Kinshasa', country: 'CD', lat: -4.4419, lon: 15.2663 },
    { name: 'Brazzaville', country: 'CG', lat: -4.2634, lon: 15.2429 },
    { name: 'Luanda', country: 'AO', lat: -8.8383, lon: 13.2344 },
    { name: 'Sao Paulo', country: 'BR', lat: -23.5505, lon: -46.6333 },
    { name: 'Brasilia', country: 'BR', lat: -15.7942, lon: -47.8822 },
    { name: 'Salvador', country: 'BR', lat: -12.9714, lon: -38.5014 },
    { name: 'Belo Horizonte', country: 'BR', lat: -19.9167, lon: -43.9345 },
    { name: 'Porto Alegre', country: 'BR', lat: -30.0346, lon: -51.2177 },
    { name: 'Santiago', country: 'CL', lat: -33.4489, lon: -70.6693 },
    { name: 'Valparaiso', country: 'CL', lat: -33.0472, lon: -71.6127 },
    { name: 'Lima', country: 'PE', lat: -12.0464, lon: -77.0428 },
    { name: 'Quito', country: 'EC', lat: -0.1807, lon: -78.4678 },
    { name: 'Guayaquil', country: 'EC', lat: -2.1900, lon: -79.8875 },
    { name: 'Bogota', country: 'CO', lat: 4.7110, lon: -74.0721 },
    { name: 'Medellin', country: 'CO', lat: 6.2476, lon: -75.5709 },
    { name: 'Caracas', country: 'VE', lat: 10.4806, lon: -66.9036 },
    { name: 'Panama City', country: 'PA', lat: 8.9833, lon: -79.5167 },
    { name: 'San Jose', country: 'CR', lat: 9.9281, lon: -84.0907 },
    { name: 'Managua', country: 'NI', lat: 12.1149, lon: -86.2362 },
    { name: 'San Salvador', country: 'SV', lat: 13.6929, lon: -89.2182 },
    { name: 'Guatemala City', country: 'GT', lat: 14.6349, lon: -90.5069 },
    { name: 'Havana', country: 'CU', lat: 23.1136, lon: -82.3666 },
    { name: 'Santo Domingo', country: 'DO', lat: 18.4861, lon: -69.9312 },
    { name: 'Port-au-Prince', country: 'HT', lat: 18.5944, lon: -72.3074 },
    { name: 'San Juan', country: 'PR', lat: 18.4655, lon: -66.1057 },
    { name: 'Guangzhou', country: 'CN', lat: 23.1291, lon: 113.2644 },
    { name: 'Shenzhen', country: 'CN', lat: 22.5431, lon: 114.0579 },
    { name: 'Chongqing', country: 'CN', lat: 29.4316, lon: 106.9123 },
    { name: 'Chengdu', country: 'CN', lat: 30.5728, lon: 104.0668 },
    { name: 'Tianjin', country: 'CN', lat: 39.3434, lon: 117.3616 },
    { name: 'Wuhan', country: 'CN', lat: 30.5928, lon: 114.3055 },
    { name: 'Xian', country: 'CN', lat: 34.3416, lon: 108.9398 },
    { name: 'Osaka', country: 'JP', lat: 34.6937, lon: 135.5022 },
    { name: 'Yokohama', country: 'JP', lat: 35.4437, lon: 139.6380 },
    { name: 'Kyoto', country: 'JP', lat: 35.0116, lon: 135.7681 },
    { name: 'Sapporo', country: 'JP', lat: 43.0618, lon: 141.3545 },
    { name: 'Busan', country: 'KR', lat: 35.1796, lon: 129.0756 },
    { name: 'Taipei', country: 'TW', lat: 25.0330, lon: 121.5654 },
    { name: 'Kaohsiung', country: 'TW', lat: 22.6273, lon: 120.3014 },
    { name: 'Manila', country: 'PH', lat: 14.5995, lon: 120.9842 },
    { name: 'Cebu', country: 'PH', lat: 10.3157, lon: 123.8854 },
    { name: 'Hanoi', country: 'VN', lat: 21.0278, lon: 105.8342 },
    { name: 'Ho Chi Minh City', country: 'VN', lat: 10.8231, lon: 106.6297 },
    { name: 'Jakarta', country: 'ID', lat: -6.2088, lon: 106.8456 },
    { name: 'Surabaya', country: 'ID', lat: -7.2575, lon: 112.7521 },
    { name: 'Bali', country: 'ID', lat: -8.3405, lon: 115.0920 },
    { name: 'Kuala Lumpur', country: 'MY', lat: 3.1390, lon: 101.6869 },
    { name: 'Penang', country: 'MY', lat: 5.4141, lon: 100.3288 },
    { name: 'Yangon', country: 'MM', lat: 16.8661, lon: 96.1951 },
    { name: 'Phnom Penh', country: 'KH', lat: 11.5564, lon: 104.9282 },
    { name: 'Vientiane', country: 'LA', lat: 17.9757, lon: 102.6331 },
    { name: 'Dhaka', country: 'BD', lat: 23.8103, lon: 90.4125 },
    { name: 'Chittagong', country: 'BD', lat: 22.3569, lon: 91.7832 },
    { name: 'Kathmandu', country: 'NP', lat: 27.7172, lon: 85.3240 },
    { name: 'Colombo', country: 'LK', lat: 6.9271, lon: 79.8612 },
    { name: 'Karachi', country: 'PK', lat: 24.8607, lon: 67.0011 },
    { name: 'Lahore', country: 'PK', lat: 31.5204, lon: 74.3587 },
    { name: 'Islamabad', country: 'PK', lat: 33.6844, lon: 73.0479 },
    { name: 'Chennai', country: 'IN', lat: 13.0827, lon: 80.2707 },
    { name: 'Kolkata', country: 'IN', lat: 22.5726, lon: 88.3639 },
    { name: 'Bangalore', country: 'IN', lat: 12.9716, lon: 77.5946 },
    { name: 'Hyderabad', country: 'IN', lat: 17.3850, lon: 78.4867 },
    { name: 'Ahmedabad', country: 'IN', lat: 23.0225, lon: 72.5714 },
    { name: 'Melbourne', country: 'AU', lat: -37.8136, lon: 144.9631 },
    { name: 'Brisbane', country: 'AU', lat: -27.4698, lon: 153.0251 },
    { name: 'Perth', country: 'AU', lat: -31.9505, lon: 115.8605 },
    { name: 'Adelaide', country: 'AU', lat: -34.9285, lon: 138.6007 },
    { name: 'Auckland', country: 'NZ', lat: -36.8509, lon: 174.7645 },
    { name: 'Wellington', country: 'NZ', lat: -41.2865, lon: 174.7762 },
    { name: 'Christchurch', country: 'NZ', lat: -43.5320, lon: 172.6306 }
];

// Mock database function for storing and retrieving data
const mockDatabase = (() => {
    const storage = window.localStorage;

    const storeData = (cityName, data) => {
        const storedData = {
            timestamp: new Date().getTime(),
            data: data
        };
        storage.setItem(`aqi_${cityName}`, JSON.stringify(storedData));
    };

    const getData = (cityName) => {
        const storedData = storage.getItem(`aqi_${cityName}`);
        if (storedData) {
            return JSON.parse(storedData);
        }
        return null;
    };

    const isDataFresh = (cityName) => {
        const storedData = getData(cityName);
        if (!storedData) return false;

        const currentTime = new Date().getTime();
        const storedTime = storedData.timestamp;
        const oneHour = 60 * 60 * 1000; // in milliseconds

        return (currentTime - storedTime) < oneHour;
    };

    return {
        storeData,
        getData,
        isDataFresh
    };
})();

const WorldAQIMap = () => {
    const location = useLocation();

    const initialIsDark = location.state?.isdark ?? true;
    const [cityData, setCityData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [lastUpdated, setLastUpdated] = useState(null);
    const [darkMode, setDarkMode] = useState(initialIsDark);
    const [hoveredCity, setHoveredCity] = useState(null);
    const apiKey = 'aac405e628f9c30a047d3de13192a7f7'; // Replace with your OpenWeatherMap API key
    const mapRef = useRef();

    // AQI categories to match the image
    const aqiCategories = [
        {
            range: [0, 50],
            level: 'Good',
            color: '#00e400',
            healthImplications: 'Air quality is satisfactory, and air pollution poses little or no risk.'
        },
        {
            range: [51, 100],
            level: 'Moderate',
            color: '#ffff00',
            healthImplications: 'Air quality is acceptable. However, there may be a risk for some people.'
        },
        {
            range: [101, 150],
            level: 'Poor',
            color: '#ff7e00',
            healthImplications: 'Members of sensitive groups may experience health effects.'
        },
        {
            range: [151, 200],
            level: 'Unhealthy',
            color: '#ff0000',
            healthImplications: 'Everyone may begin to experience health effects.'
        },
        {
            range: [201, 300],
            level: 'Severe',
            color: '#8f3f97',
            healthImplications: 'Health alert: Health risk increases for everyone.'
        },
        {
            range: [301, 500],
            level: 'Hazardous',
            color: '#7e0023',
            healthImplications: 'Health warning of emergency conditions: everyone more likely to be affected.'
        }
    ];

    // Toggle theme
    const toggleDarkMode = () => {
        const newDarkMode = !darkMode;
        setDarkMode(newDarkMode);
        localStorage.setItem('darkMode', newDarkMode);
        document.body.className = newDarkMode ? 'dark-mode' : 'light-mode';
    };

    useEffect(() => {
        fetchAllCityData();
        // Set up auto-refresh every hour
        const refreshInterval = setInterval(() => {
            fetchAllCityData();
        }, 60 * 60 * 1000); // 1 hour in milliseconds

        return () => clearInterval(refreshInterval);
    }, []);

    useEffect(() => {
        // Apply map style based on dark mode
        const mapElement = document.querySelector('.leaflet-container');
        if (mapElement) {
            if (darkMode) {
                mapElement.style.filter = 'brightness(0.8) saturate(0.8)';
            } else {
                mapElement.style.filter = 'none';
            }
        }
    }, [darkMode]);

    const fetchAllCityData = async () => {
        setLoading(true);
        const newCityData = [];

        // Process cities in batches to avoid overwhelming the API
        const batchSize = 5;
        for (let i = 0; i < majorCities.length; i += batchSize) {
            const batch = majorCities.slice(i, i + batchSize);
            await Promise.all(batch.map(city => fetchCityAQIData(city, newCityData)));
            // Small delay between batches
            if (i + batchSize < majorCities.length) {
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }

        setCityData(newCityData);
        setLastUpdated(new Date());
        setLoading(false);
    };

    const fetchCityAQIData = async (city, dataArray) => {
        // Check if we have fresh data in the database
        if (mockDatabase.isDataFresh(city.name)) {
            const storedData = mockDatabase.getData(city.name);
            dataArray.push({
                ...city,
                ...storedData.data,
                fromCache: true
            });
            return;
        }

        // If not, fetch from API
        try {
            const response = await fetch(
                `https://api.openweathermap.org/data/2.5/air_pollution?lat=${city.lat}&lon=${city.lon}&appid=${apiKey}`
            );

            if (!response.ok) {
                throw new Error(`Failed to fetch data for ${city.name}`);
            }

            const data = await response.json();

            if (data && data.list && data.list.length > 0) {
                const pollutants = data.list[0].components;
                const aqi = calculateAQI(pollutants);
                const aqiCategory = getAQICategory(aqi);

                const cityAQIData = {
                    ...city,
                    pollutants,
                    aqi,
                    aqiCategory,
                    fromCache: false
                };

                // Store in database
                mockDatabase.storeData(city.name, {
                    pollutants,
                    aqi,
                    aqiCategory
                });

                dataArray.push(cityAQIData);
            }
        } catch (error) {
            console.error(`Error fetching data for ${city.name}:`, error);
            // Try to get any data we might have in cache, even if it's old
            const storedData = mockDatabase.getData(city.name);
            if (storedData) {
                dataArray.push({
                    ...city,
                    ...storedData.data,
                    fromCache: true,
                    dataStale: true
                });
            }
        }
    };

    const getMarkerIcon = (city) => {
        const isHovered = hoveredCity && hoveredCity.name === city.name;
        
        // Calculate pin size based on PM2.5 - more reasonable range
        // Base size is smaller, max size is limited to prevent overcrowding
        const pm25Value = city.pollutants?.pm2_5 || 0;
        let sizeMultiplier = 1;
        
        // Scale the marker size based on PM2.5 values, but keep it reasonable
        if (pm25Value < 12) {
            sizeMultiplier = 0.8; // Good AQI
        } else if (pm25Value < 35.4) {
            sizeMultiplier = 0.9; // Moderate AQI
        } else if (pm25Value < 55.4) {
            sizeMultiplier = 1.0; // Poor AQI
        } else if (pm25Value < 150.4) {
            sizeMultiplier = 1.1; // Unhealthy
        } else if (pm25Value < 250.4) {
            sizeMultiplier = 1.2; // Severe
        } else {
            sizeMultiplier = 1.3; // Hazardous
        }
        
        // Increase size slightly if hovered
        if (isHovered) {
            sizeMultiplier *= 1.3;
        }
        
        // Determine color based on AQI category
        const color = city.aqiCategory?.color || '#999';
        
        // SVG map pin
        const pinSVG = `
            <svg xmlns="http://www.w3.org/2000/svg" 
                 viewBox="0 0 24 36" 
                 width="${24 * sizeMultiplier}" 
                 height="${36 * sizeMultiplier}">
                <path fill="${color}" 
                      stroke="#FFFFFF" 
                      stroke-width="1.5" 
                      d="M12 0C5.383 0 0 5.383 0 12c0 3.912 2.167 7.417 5.583 9.167L12 36l6.417-14.833C21.833 19.417 24 15.912 24 12c0-6.617-5.383-12-12-12z">
                </path>
                ${isHovered ? `<circle cx="12" cy="12" r="6" fill="white" fill-opacity="0.6"/>` : ''}
            </svg>
        `;

        return L.divIcon({
            html: `
                <div class="map-pin-container">
                    ${pinSVG}
                    ${isHovered ? `
                        <div class="hover-tooltip">
                            <strong>${city.name}</strong>
                            <span class="tooltip-aqi">AQI: ${city.aqi} - ${city.aqiCategory?.level}</span>
                        </div>
                    ` : ''}
                </div>
            `,
            className: 'map-pin-icon',
            iconSize: [24 * sizeMultiplier, 36 * sizeMultiplier],
            iconAnchor: [12 * sizeMultiplier, 36 * sizeMultiplier], // Bottom middle of the pin
            popupAnchor: [0, -36 * sizeMultiplier] // Above the pin
        });
    };

    // Marker event handlers
    const handleMarkerMouseOver = (city) => {
        setHoveredCity(city);
    };

    const handleMarkerMouseOut = () => {
        setHoveredCity(null);
    };

    // Create interactive markers
    const createInteractiveMarker = (city, index) => {
        return (
            <Marker
                key={`${city.name}-${index}`}
                position={[city.lat, city.lon]}
                icon={getMarkerIcon(city)}
                eventHandlers={{
                    mouseover: () => handleMarkerMouseOver(city),
                    mouseout: () => handleMarkerMouseOut(),
                }}
            >
                <Popup className="city-popup">
                    <div className="popup-header" style={{ backgroundColor: city.aqiCategory?.color }}>
                        <h3>{city.name}, {city.country}</h3>
                        <div className="aqi-badge">
                            <span className="aqi-value">{city.aqi}</span>
                            <span className="aqi-label">{city.aqiCategory?.level}</span>
                        </div>
                    </div>

                    <div className="pollutant-details">
                        <h4>Pollutants</h4>
                        <div className="pollutant-grid">
                            <div className="pollutant-item">
                                <span className="pollutant-name">PM2.5</span>
                                <span className="pollutant-value">{city.pollutants?.pm2_5?.toFixed(1)} μg/m³</span>
                            </div>
                            <div className="pollutant-item">
                                <span className="pollutant-name">PM10</span>
                                <span className="pollutant-value">{city.pollutants?.pm10?.toFixed(1)} μg/m³</span>
                            </div>
                            <div className="pollutant-item">
                                <span className="pollutant-name">O3</span>
                                <span className="pollutant-value">{city.pollutants?.o3?.toFixed(1)} μg/m³</span>
                            </div>
                            <div className="pollutant-item">
                                <span className="pollutant-name">NO2</span>
                                <span className="pollutant-value">{city.pollutants?.no2?.toFixed(1)} μg/m³</span>
                            </div>
                            <div className="pollutant-item">
                                <span className="pollutant-name">SO2</span>
                                <span className="pollutant-value">{city.pollutants?.so2?.toFixed(1)} μg/m³</span>
                            </div>
                            <div className="pollutant-item">
                                <span className="pollutant-name">CO</span>
                                <span className="pollutant-value">{city.pollutants?.co?.toFixed(1)} μg/m³</span>
                            </div>
                        </div>
                    </div>

                    {city.fromCache && (
                        <p className="cache-notice">
                            {city.dataStale ? "Using cached data (stale)" : "Using cached data (< 1 hour old)"}
                        </p>
                    )}
                </Popup>
            </Marker>
        );
    };

    return (
        <div className={darkMode ? 'dark-theme' : 'light-theme'}>
            <TopBar isDark={darkMode} toggleDarkMode={toggleDarkMode} />
            <div className="world-aqi-map-container">
                <div className="map-header">
                    <h1>World Air Quality Index Map</h1>
                    <div className="map-info">
                        {lastUpdated && (
                            <div className="last-updated">
                                <i className="fas fa-clock"></i>
                                <span>Last updated: {lastUpdated.toLocaleString()}</span>
                            </div>
                        )}
                        <button
                            className="refresh-button"
                            onClick={fetchAllCityData}
                            disabled={loading}
                        >
                            <i className="fas fa-sync-alt"></i> Refresh Data
                        </button>
                    </div>
                </div>

                <div className="map-container">
                    {loading && (
                        <div className="loading-overlay">
                            <div className="loading-spinner"></div>
                            <p>Loading AQI data...</p>
                        </div>
                    )}
                    
                    <MapContainer
                        center={[20, 0]}
                        zoom={2}
                        minZoom={2}
                        style={{ height: '600px', width: '100%' }}
                        ref={mapRef}
                        zoomControl={false}
                        className={darkMode ? 'dark-map' : 'light-map'}
                    >
                        <TileLayer
                            url={darkMode
                                ? "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                                : "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                            }
                            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                        />
                        <ZoomControl position="bottomright" />

                        {cityData.map((city, index) => createInteractiveMarker(city, index))}
                    </MapContainer>
                </div>

                {/* AQI Legend styled according to the image */}
                <div className="aqi-legend-container">
                    <div className="aqi-legend-labels">
                        {aqiCategories.map((category, index) => (
                            <div key={index} className="aqi-legend-label">
                                {category.level}
                            </div>
                        ))}
                    </div>
                    <div className="aqi-legend-bar">
                        {aqiCategories.map((category, index) => (
                            <div
                                key={index}
                                className="aqi-legend-segment"
                                style={{ backgroundColor: category.color }}
                            ></div>
                        ))}
                    </div>
                    <div className="aqi-legend-values">
                        <span>0</span>
                        <span>50</span>
                        <span>100</span>
                        <span>150</span>
                        <span>200</span>
                        <span>300</span>
                        <span>301+</span>
                    </div>
                </div>

                <div className="pin-size-legend">
                        <h4>Map Pins Info</h4>
                        <div className="pin-size-examples">
                            <div className="pin-example">
                                <div className="example-pin small" style={{ backgroundColor: aqiCategories[0].color }}></div>
                                <span>Good PM2.5</span>
                            </div>
                            <div className="pin-example">
                                <div className="example-pin medium" style={{ backgroundColor: aqiCategories[0].color }}></div>
                                <span>Moderate PM2.5</span>
                            </div>
                            <div className="pin-example">
                                <div className="example-pin large" style={{ backgroundColor: aqiCategories[0].color }}></div>
                                <span>Hazardous PM2.5</span>
                            </div>
                        </div>
                    </div>
            </div>
            <Footer />
        </div>
    );
};

export default WorldAQIMap;