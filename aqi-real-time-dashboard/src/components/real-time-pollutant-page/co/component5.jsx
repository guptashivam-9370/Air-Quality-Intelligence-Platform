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
      title: "Fatigue",
      description: "Feeling tiredness or lack of energy.",
      imageSrc: "/co/c5i1.webp"
    },
    {
      title: "Chest Pain",
      description: "Discomfort or pain in the chest area.",
      imageSrc: "/co/c5i2.webp"
    },
    {
      title: "Confusion",
      description: "Frequent difficulty in thinking clearly.",
      imageSrc: "/co/c5i3.webp"
    },
    {
      title: "Flu-like Symptoms",
      description: "Symptoms like headache and nausea.",
      imageSrc: "/co/c5i4.webp"
    },
    {
      title: "Weakness",
      description: "Frequent weakness or lack of strength.",
      imageSrc: "/co/c5i5.webp"
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