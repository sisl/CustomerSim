
cd src

Rscript kdd98_preprocess.R
python kdd98_initial_snapshot.py
python kdd98_propagate_regressor.py
python kdd98_propagate_classifier.py
python kdd98_simulate.py

python vs_preprocess_category.py
python vs_propagate_regressor_category.py
python vs_simulate.py

cd ..