# Creativity Scoring in Soccer Using Hybrid Modeling

This project introduces a hybrid machine learning pipeline to predict soccer player creativity scores by combining rule-based soccer skill moves/action detection, sequence modeling, and spatial graph modeling.

## Project Structure
- **creativity_score_pipeline.py**: Full end-to-end Python code to preprocess match event data, detect soccer skill moves/actions, build player-action graphs, train Transformer and GNN models, and fuse outputs into the hybrid CreaScoreNet model.
- **top10_creativity_scores.png**: Visualization of top-ranked players by predicted creativity score.
- **statsbomb_json/**: Folder containing sample StatsBomb match event JSON files used for training and evaluation.
- **CS_Demo_Day_Poster.pdf**: Poster presented at the 2025 GSU Computer Science Demo Day summarizing the project.
- **Creativity_Scoring_Research_Paper.pdf**: Full research paper detailing problem statement, methodology, technical approach, evaluation, and results.

## Key Features
- **Soccer Skill Moves/Action Detection**: Rule-based scoring for nutmegs, pressured dribbles, deceptive passes, 1v1 wins, and more.
- **Sequential Modeling**: Transformer Encoder used to capture temporal dependencies of soccer skill moves/action sequences.
- **Graph-Based Modeling**: Graph Neural Network (GNN) used to capture spatial relationships between player actions.
- **Hybrid Model**: CreaScoreNet fuses GNN and Transformer outputs for final creativity prediction.
- **Evaluation**: Validated on StatsBomb Open Data, SofaScore, and FutMob player event datasets. Visualization of player rankings included.

## How to Run
1. Upload the StatsBomb match event JSON files (`statsbomb_json/` folder).
2. Run `creativity_score_pipeline.py` in a Google Colab or local Python environment.
3. View the generated creativity scores and bar chart outputs.

## Dependencies
- PyTorch
- torch-geometric
- pandas
- matplotlib
- scikit-learn

Install using:
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pandas matplotlib scikit-learn

