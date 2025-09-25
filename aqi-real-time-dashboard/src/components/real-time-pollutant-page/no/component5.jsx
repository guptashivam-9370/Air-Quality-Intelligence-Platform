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

  const impactsNO = [
    {
      title: "Respiratory Irritation",
      description: "Exposure to nitrogen oxides can cause inflammation and irritation in the respiratory tract.",
      imageSrc: "/no/c5i1.webp" // Replace with your actual image paths
    },
    {
      title: "Aggravated Asthma",
      description: "NO can exacerbate asthma symptoms and trigger asthma attacks in sensitive individuals.",
      imageSrc: "/no/c5i2.webp"
    },
    {
      title: "Reduced Lung Function",
      description: "Long-term exposure may lead to decreased lung capacity and impaired breathing.",
      imageSrc: "/no/c5i3.webp"
    },
    {
      title: "Increased Respiratory Infections",
      description: "Higher levels of nitrogen oxides are associated with a greater risk of respiratory infections.",
      imageSrc: "/no/c5i4.webp"
    },
    {
      title: "Environmental Impact",
      description: "Nitrogen oxides contribute to smog and acid rain, affecting ecosystems and vegetation.",
      imageSrc: "/no/c5i5.webp"
    }
  ];

  return (
    <div className="health-impacts-container">
      <h2>Short-Term NO Exposure Impacts</h2>

      <div className="impacts-grid">
        {impactsNO.map((impact, index) => (
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