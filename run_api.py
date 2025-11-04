import subprocess

if __name__ == "__main__":
    subprocess.run(
        ["uv", "run", "uvicorn", "main:app", "--reload", "--port", "8000"],
        cwd="api"
    )

# uv run streamlit run ui.py