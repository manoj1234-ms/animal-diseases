import joblib

stage1 = joblib.load('models/stage1_pipeline.pkl')
print('=== Stage1 Pipeline ===')
print('steps:', list(stage1.named_steps.keys()))
pre = stage1.named_steps['preprocessor']
print('preprocessor.get_feature_names_out len:', len(pre.get_feature_names_out()))
print('preprocessor.get_feature_names_out:', pre.get_feature_names_out())
model = stage1.named_steps['model']
print('model type:', type(model))
print('model.n_features_in_:', getattr(model, 'n_features_in_', None))
try:
    booster = model.get_booster()
    print('booster.feature_names:', booster.feature_names)
    print('booster.num_features():', booster.num_features())
except Exception as e:
    print('booster info error:', e)

print('\n=== Stage2 Models ===')
stage2 = joblib.load('models/stage2_models.pkl')
print('categories:', list(stage2.keys()))
for cat, pipe in stage2.items():
    print('\nCategory:', cat)
    print(' pipeline steps:', list(pipe.named_steps.keys()))
    pre2 = pipe.named_steps.get('preprocessor')
    if pre2 is not None:
        try:
            print(' preprocessor.get_feature_names_out len:', len(pre2.get_feature_names_out()))
        except Exception:
            print(' preprocessor.get_feature_names_out: not available')
    m = pipe.named_steps.get('model')
    print(' model type:', type(m))
    print(' model.n_features_in_:', getattr(m, 'n_features_in_', None))
    try:
        b = m.get_booster()
        print('  booster.num_features():', b.num_features())
    except Exception as e:
        print('  booster info error:', e)

print('\nDone')
