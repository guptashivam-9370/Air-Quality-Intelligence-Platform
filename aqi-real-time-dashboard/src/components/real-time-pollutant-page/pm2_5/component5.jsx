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
      title: "Irritation in Eyes",
      description: "Redness, itching, and discomfort in your eyes",
      imageSrc: "/pm2_5/c5i1.webp" // Replace with your actual image paths
    },
    {
      title: "Headaches",
      description: "Frequent or intense headaches.",
      imageSrc: "/pm2_5/c5i2.webp"
    },
    {
      title: "Fatigue",
      description: "Feeling unusually tired or weak.",
      imageSrc: "/pm2_5/c5i3.webp"
    },
    {
      title: "Aggravated asthma",
      description: "Increased asthma attacks and symptoms.",
      imageSrc: "/pm2_5/c5i4.webp"
    },
    {
      title: "Breathing problems",
      description: "Coughing, wheezing, and shortness of breath.",
      imageSrc: "/pm2_5/c5i5.webp"
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