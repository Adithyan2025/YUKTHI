"""Streamlit Cloud entry point."""

# The application lives in app.py; this wrapper keeps the conventional
# Streamlit Cloud filename available for deployment configuration.
exec(compile(open("app.py", encoding="utf-8").read(), "app.py", "exec"))
