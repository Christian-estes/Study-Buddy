import subprocess

if __name__ == "__main__":
    subprocess.run(
        ["uv", "run", "streamlit", "run", "ui.py", "--server.port", "8502"],
        cwd="frontend"  
    )

# uv run streamlit run ui.py