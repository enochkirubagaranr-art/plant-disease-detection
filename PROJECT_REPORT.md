# Project Evaluation Report

## 1. Code Execution and Prediction Result

The model was loaded and evaluated using the project evaluation script `eval_and_report.py`. Below are sample prediction results on validation images.

- File: C:\Users\Enoch\Desktop\FINAL\dataset\Potato_Early_Blight\001187a0-57ab-4329-baff-e7246a9edeb0___RS_Early.B 8178.JPG\n  - True label: **Potato_Early_Blight**\n  - Predicted: **Potato_Early_Blight** (confidence: 99.89%)\n
- File: C:\Users\Enoch\Desktop\FINAL\dataset\Potato_Early_Blight\002a55fb-7a3d-4a3a-aca8-ce2d5ebc6925___RS_Early.B 8170.JPG\n  - True label: **Potato_Early_Blight**\n  - Predicted: **Potato_Early_Blight** (confidence: 97.73%)\n
- File: C:\Users\Enoch\Desktop\FINAL\dataset\Potato_Early_Blight\009c8c31-f22d-4ffd-8f16-189c6f06c577___RS_Early.B 7885.JPG\n  - True label: **Potato_Early_Blight**\n  - Predicted: **Potato_Early_Blight** (confidence: 99.99%)\n
- File: C:\Users\Enoch\Desktop\FINAL\dataset\Potato_Early_Blight\00d8f10f-5038-4e0f-bb58-0b885ddc0cc5___RS_Early.B 8722.JPG\n  - True label: **Potato_Early_Blight**\n  - Predicted: **Potato_Early_Blight** (confidence: 92.52%)\n
- File: C:\Users\Enoch\Desktop\FINAL\dataset\Potato_Early_Blight\0182e991-97f0-4805-a1f7-6e1b4306d518___RS_Early.B 7015.JPG\n  - True label: **Potato_Early_Blight**\n  - Predicted: **Potato_Early_Blight** (confidence: 99.64%)\n
## 2. Testing Accuracy Output

- Validation samples: 1198\n- Validation accuracy: **94.57%**\n
### Confusion Matrix (rows=true class, cols=predicted class)

```
193 0 7 0 0 0
0 194 4 0 2 0
0 2 195 1 0 2
1 0 0 176 4 19
0 0 0 0 198 1
0 1 2 17 2 177
```

### Per-class accuracy

- Potato_Early_Blight: 193/200 = **96.50%**\n- Potato_Healthy: 194/200 = **97.00%**\n- Potato_Late_Blight: 195/200 = **97.50%**\n- Tomato_Early_Blight: 176/200 = **88.00%**\n- Tomato_Healthy: 198/199 = **99.50%**\n- Tomato_Late_Blight: 177/199 = **88.94%**\n