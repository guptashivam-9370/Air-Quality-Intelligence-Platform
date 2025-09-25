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
      title: "Coughing",
      description: "Regular cough due to airways irritations.",
      imageSrc: "/no2/c5i1.webp"
    },
    {
      title: "Wheezing",
      description: "More whistling sound while breathing.",
      imageSrc: "/no2/c5i2.webp"
    },
    {
      title: "Difficulty in Breathing",
      description: "Shortness of breath and reduced lung function.",
      imageSrc: "/no2/c5i3.webp"
    },
    {
      title: "Reduce Smelling Ability",
      description: "Decreased sense of smell due to irritation.",
      imageSrc: "/no2/c5i4.webp"
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