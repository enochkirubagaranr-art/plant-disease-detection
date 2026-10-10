# Publishing the Plant Disease Streamlit App

## Recommended route: Streamlit Community Cloud

1. Push this repository to GitHub.
2. Make sure the root contains:
   - `streamlit_app.py`
   - `FINAL_MODEL.h5`
   - `class_names.json`
   - `requirements.txt`
3. Open https://share.streamlit.io and sign in with GitHub.
4. Create a new app and select this repository and branch.
5. Set the app file to `streamlit_app.py`.
6. Deploy.

## Notes

- `FINAL_MODEL.h5` is about 13.8 MB, so it can be stored in GitHub normally.
- If you prefer, you can deploy with other cloud hosts like Render, Railway, or an Azure Web App using the same `requirements.txt`.

## Local run command

```bash
c:/Users/Enoch/Desktop/FINAL/venv/Scripts/python.exe -m streamlit run streamlit_app.py
```

## If you want to use `app.py` instead

- Change the app file to `app.py` when configuring deployment.
- Ensure `plant_model.h5` and `class_names.json` are also included in the repo.
