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
      title: "Chest Pain",
      description: "Discomfort or pain in the chest due to respiratory stress.",
      imageSrc: "/o3/c5i1.webp"
    },
    {
      title: "Breathing Shortness",
      description: "Difficulty in breathing or feeling of breathlessness.",
      imageSrc: "/o3/c5i2.webp"
    },
    {
      title: "Respiratory Irritation",
      description: "Inflammation and irritation of the respiratory tract.",
      imageSrc: "/o3/c5i3.webp"
    },
    {
      title: "Wheezing and Coughing",
      description: "Whistling sound and persistent cough.",
      imageSrc: "/o3/c5i4.webp"
    },
    {
      title: "Trigger Asthma Attacks",
      description: "Aggravation of asthma symptoms.",
      imageSrc: "/o3/c5i5.webp"
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