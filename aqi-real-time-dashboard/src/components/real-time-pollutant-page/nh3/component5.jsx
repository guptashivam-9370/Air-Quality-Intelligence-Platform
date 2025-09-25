import React from 'react';
import './Component5.css'; // Make sure to create this CSS file

const HealthImpactCard = ({ imageSrc, title, description }) => {
  return (
    <div className="health-impact-card">
      <div className="image-container">
        <img src={imageSrc} alt={title} />
      </div>
      <div className="content">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
};

const Component5 = () => {
  const impactsNH3 = [
    {
      title: "Respiratory Issues",
      description: "Exposure to ammonia can cause irritation and damage to the respiratory system.",
      imageSrc: "/nh3/c5i1.webp" // Replace with your actual image paths
    },
    {
      title: "Eye Irritation",
      description: "Ammonia vapors can irritate the eyes, leading to redness and discomfort.",
      imageSrc: "/nh3/c5i2.webp"
    },
    {
      title: "Environmental Damage",
      description: "Ammonia contributes to nutrient pollution, which can harm aquatic life and ecosystems.",
      imageSrc: "/nh3/c5i3.webp"
    },
    {
      title: "Soil Acidification",
      description: "When deposited, ammonia can change soil pH, impacting plant growth and soil health.",
      imageSrc: "/nh3/c5i4.webp"
    },
    {
      title: "Increased Smog Formation",
      description: "Ammonia plays a role in secondary particulate matter formation, contributing to smog.",
      imageSrc: "/nh3/c5i5.webp"
    }
  ];

  return (
    <div className="health-impacts-container">
      <h2>Short-Term NH3 Exposure Impacts</h2>

      <div className="impacts-grid">
        {impactsNH3.map((impact, index) => (
          <HealthImpactCard
            key={index}
            title={impact.title}
            description={impact.description}
            imageSrc={impact.imageSrc}
          />
        ))}
      </div>
    </div>
  );
};

export default Component5;