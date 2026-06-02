# ── Run this cell in Colab after training all 3 models ──────
# This saves all trained models into the correct directory structure

import os
import joblib

os.makedirs('clinical_note_intelligence/models', exist_ok=True)
os.makedirs('clinical_note_intelligence/data', exist_ok=True)

# Save Model 3 artifacts
joblib.dump(model,        'clinical_note_intelligence/models/risk_model.pkl')
joblib.dump(risk_encoder, 'clinical_note_intelligence/models/risk_encoder.pkl')
joblib.dump(ohe,          'clinical_note_intelligence/models/ohe_encoder.pkl')
joblib.dump(encoder,      'clinical_note_intelligence/models/sentence_encoder.pkl')

# Save final dataset
df.to_csv('clinical_note_intelligence/data/mtsamples_final.csv', index=False)

print("✅ All models saved!")
print("\nDirectory structure:")
for root, dirs, files in os.walk('clinical_note_intelligence'):
    level    = root.replace('clinical_note_intelligence', '').count(os.sep)
    indent   = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        size = os.path.getsize(os.path.join(root, file))
        print(f'{subindent}{file}  ({size/1024:.1f} KB)')
