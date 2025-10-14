interface Props {
  data: any;
}

const AnalysisResult: React.FC<Props> = ({ data }) => {
  if (!data) return null;

  return (
    <div className="analysis">
      <h2>🌾 Crop Health Analysis</h2>
      <p><strong>NDVI:</strong> {data.ndvi.toFixed(3)}</p>
      <p><strong>Health Status:</strong> {data.health}</p>
      <p><strong>Recommendation:</strong> {data.recommendation}</p>
    </div>
  );
};

export default AnalysisResult;
