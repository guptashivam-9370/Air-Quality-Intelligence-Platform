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
  const impacts = [
    {
      title: "Allergies",
      description: "Airborne allergens causing sneezing and itching.",
      imageSrc: "/pm10/c5i1.webp"
    },
    {
      title: "Cough and Runny Nose",
      description: "Respiratory irritation leading to coughing and nasal discharge.",
      imageSrc: "/pm10/c5i2.webp"
    },
    {
      title: "Eyes Irritations",
      description: "Redness and discomfort in the eyes.",
      imageSrc: "/pm10/c5i3.webp"
    },
    {
      title: "Chest Tightness",
      description: "Constriction feeling in the chest.",
      imageSrc: "/pm10/c5i4.webp"
    },
    {
      title: "Breathing Difficulty",
      description: "Trouble breathing or discomfort in the chest.",
      imageSrc: "/pm10/c5i5.webp"
    }
  ];


  return (
    <div className="health-impacts-container">
      <h2>Short-Term PM2.5 Exposure Impacts</h2>

      <div className="impacts-grid">
        {impacts.map((impact, index) => (
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