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
      title: "Nose Irritation",
      description: "Discomfort in the nasal passages.",
      imageSrc: "/so2/c5i1.webp"
    },
    {
      title: "Throat and Lung Irritation",
      description: "Inflammation and pain in the throat.",
      imageSrc: "/so2/c5i2.webp"
    },
    {
      title: "Coughing and Phlegm",
      description: "Persistent cough and mucus production.",
      imageSrc: "/so2/c5i3.webp"
    },
    {
      title: "Asthma Attacks",
      description: "Triggering of asthma symptoms or existing asthma.",
      imageSrc: "/so2/c5i4.webp"
    },
    {
      title: "Wheezing",
      description: "High-pitched whistling sound when breathing.",
      imageSrc: "/so2/c5i5.webp"
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