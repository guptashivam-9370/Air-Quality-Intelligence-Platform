import React from 'react';

const AmmoniaSourcesComponent = () => {
  const sources = [
    {
      id: 1,
      title: 'Agricultural Fertilizers',
      description: 'Ammonia is released during the application of nitrogen-based fertilizers.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M4 20v-2a4 4 0 014-4h8a4 4 0 014 4v2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M8 12V4h8v8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 2,
      title: 'Animal Waste',
      description: 'Manure from livestock emits ammonia into the atmosphere.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M4 12l1.5 2 2-3 2 4 2-3 1.5 2 3-5 2.5 4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 3,
      title: 'Sewage Treatment',
      description: 'NH₃ is produced during the breakdown of organic waste in wastewater.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M3 3h18v18H3V3z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M7 7h10v10H7V7z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 4,
      title: 'Composting & Decomposition',
      description: 'Biological breakdown of organic matter releases ammonia into the air.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <circle cx="12" cy="12" r="9" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M12 7v5l3 3" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 5,
      title: 'Industrial Refrigerants',
      description: 'Used in some cooling systems and may leak during use or disposal.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M8 2v20M16 2v20M4 8h16M4 16h16" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    }
  ];

  return (
    <div className="nh3-container">
      <div className="nh3-content">
        <div className="nh3-header">
          <h1>Uncovering the Sources of Ammonia (NH₃):</h1>
          <h2>Where Does It Come From?</h2>
        </div>

        <div className="nh3-sources-grid">
          {sources.map(source => (
            <div key={source.id} className="nh3-source-card">
              <div className="nh3-source-content">
                <div className="nh3-source-icon">{source.icon}</div>
                <div className="nh3-source-text">
                  <h3>{source.title}</h3>
                  <p>{source.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <style jsx>{`
        .nh3-container {
          position: relative;
          width: 100%;
          background-color: rgba(0, 0, 0, 0.5);
          background-image: url('https://www.bronkhorst.com/getmedia/bad33267-1009-442d-8255-6aed64f0b6d4/an65-exhaust-fumes.jpg?width=630&hash=k3cFJiEx9BdmIrbbECu1YRUsOmCGG4CtlZs2zpJWmuo%3d');
          background-size: cover;
          background-position: center;
          background-blend-mode: overlay;
        }

        .nh3-content {
          max-width: 1200px;
          margin: 0 auto;
          padding: 48px 16px;
        }

        .nh3-header {
          max-width: 900px;
          margin: 0 auto 48px;
          text-align: center;
        }

        .nh3-header h1 {
          font-size: 32px;
          font-weight: bold;
          color: white;
          margin-bottom: 16px;
        }

        .nh3-header h2 {
          font-size: 28px;
          font-weight: bold;
          color: white;
        }

        .nh3-sources-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 24px;
        }

        .nh3-source-card {
          background-color: rgba(51, 51, 51, 0.7);
          border-radius: 8px;
          padding: 16px;
          backdrop-filter: blur(4px);
          border: 1px solid rgba(119, 119, 119, 0.7);
        }

        .nh3-source-content {
          display: flex;
          align-items: flex-start;
        }

        .nh3-source-icon {
          color: white;
          margin-right: 12px;
          flex-shrink: 0;
        }

        .nh3-source-icon svg {
          width: 24px;
          height: 24px;
        }

        .nh3-source-text h3 {
          color: white;
          font-weight: bold;
          margin-bottom: 4px;
          font-size: 16px;
        }

        .nh3-source-text p {
          color: rgb(209, 213, 219);
          font-size: 14px;
        }

        @media (max-width: 768px) {
          .nh3-header h1 {
            font-size: 24px;
          }

          .nh3-header h2 {
            font-size: 20px;
          }
        }
      `}</style>
    </div>
  );
};

export default AmmoniaSourcesComponent;
