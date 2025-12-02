# Streamlit Cloud Deployment Guide

## Overview
This application is ready for deployment to Streamlit Cloud. It includes all necessary error handling and configuration for cloud deployment.

## Key Features for Cloud Deployment

### ✅ What Works Well
1. **On-demand data generation**: Users can generate synthetic datasets directly in the app
2. **On-demand model training**: Models can be trained directly in the app
3. **Error handling**: Graceful handling of missing files with helpful error messages
4. **File persistence**: Files are written to the app's file system (ephemeral between deployments)
5. **Caching**: Streamlit caching optimizes performance

### ⚠️ Important Considerations

#### File Persistence
- **Generated CSV files** and **trained models** are saved to the app's file system
- These files **will NOT persist** between Streamlit Cloud deployments or app restarts
- Users need to regenerate data and retrain models after each deployment/restart
- This is acceptable for an MVP/demo application

#### Memory & Performance
- Model training can be memory-intensive
- Training time depends on dataset size (configurable in `config.py`)
- Streamlit Cloud has timeout limits (typically 30 seconds for free tier)
- If training times out, consider reducing dataset sizes in `config.py`

#### Dependencies
- All dependencies are listed in `requirements.txt`
- Streamlit Cloud will automatically install these on deployment

## Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `aero_ai_platform/ui/app.py`
   - Click "Deploy"

3. **First-time Setup**
   - After deployment, click "Generate synthetic CSV files" button
   - Then click "Train all models" button
   - The app is now ready to use!

## Configuration

### Adjusting Dataset Sizes
If you encounter timeout issues, edit `aero_ai_platform/config.py`:

```python
class FlightConfig:
    N_FLIGHTS = 500  # Reduce if timeout occurs
    N_POINTS_PER_FLIGHT = 300  # Reduce if timeout occurs

class EngineConfig:
    N_ENGINES = 50  # Reduce if timeout occurs
    N_CYCLES_PER_ENGINE = 200  # Reduce if timeout occurs

class SHMConfig:
    N_SENSORS = 20  # Reduce if timeout occurs
    N_SAMPLES_PER_SENSOR = 5000  # Reduce if timeout occurs
```

### Adjusting Model Complexity
If model training times out, edit the model files to reduce complexity:
- Reduce `n_estimators` in RandomForest models
- Reduce `max_depth` if specified

## Troubleshooting

### "File not found" errors
- **Solution**: Click "Generate synthetic CSV files" first, then "Train all models"

### Training timeout
- **Solution**: Reduce dataset sizes or model complexity (see Configuration above)

### Import errors
- **Solution**: Ensure `requirements.txt` includes all dependencies
- Check that the file path in Streamlit Cloud is correct: `aero_ai_platform/ui/app.py`

### Memory errors
- **Solution**: Reduce dataset sizes in `config.py`

## File Structure for Streamlit Cloud
```
.
├── requirements.txt          # Root-level requirements (required)
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── aero_ai_platform/
    ├── ui/
    │   └── app.py           # Main Streamlit app
    ├── config.py
    ├── data_generation/
    ├── features/
    ├── models/
    └── ...
```

## Notes
- The app uses relative paths, so it works in any environment
- All directories are created automatically if they don't exist
- Error messages guide users to generate data/train models when needed

