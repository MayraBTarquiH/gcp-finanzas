CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #F3F5F7; }
h1, h2, h3 { font-family: 'IBM Plex Serif', serif !important; color: #12263A; letter-spacing: -0.01em; }
.block-container { padding-top: 2.2rem; max-width: 1150px; }
#MainMenu, footer { visibility: hidden; }

[data-testid="stSidebar"] { background: #12263A; }
[data-testid="stSidebar"] * { color: #DCE6EF; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: .55rem .8rem; border-radius: 8px; width: 100%; cursor: pointer;
  border-left: 3px solid transparent; transition: background .15s; }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.07); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: rgba(15,139,141,.30); border-left-color: #3FC1C3; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { font-weight: 600; color: #fff; }
.marca { font-family: 'IBM Plex Serif', serif; font-size: 1.25rem; font-weight: 600; color: #fff;
  line-height: 1.25; margin: .2rem 0 .2rem; }
.marca-sub { font-size: .82rem; color: #9FB3C6; margin-bottom: 1.1rem; }
.estado { margin-top: 1.2rem; padding: .7rem .8rem; border-radius: 8px; background: rgba(255,255,255,.06);
  font-size: .85rem; }

.hero { background: #12263A; border-radius: 14px; padding: 2.2rem 2.2rem 1.8rem; margin-bottom: 1.4rem; }
.hero h1 { color: #fff !important; font-size: 2.1rem; margin: 0 0 .6rem; line-height: 1.2; }
.hero p { color: #B9CBDB; font-size: 1.05rem; margin: 0; max-width: 46rem; }

[data-testid="stMetric"] { background: #fff; border: 1px solid #D9E0E6; border-radius: 10px; padding: .8rem 1rem; }
.nota { background: #fff; border-left: 4px solid #0F8B8D; border-radius: 6px; padding: .8rem 1rem; margin: .6rem 0 1rem; }
.nota.aviso { border-left-color: #C98A1B; }
.ficha { background: #fff; border: 1px solid #D9E0E6; border-radius: 10px; padding: 1rem 1.1rem; height: 100%; }
.ficha h4 { margin: 0 0 .3rem; font-family: 'IBM Plex Serif', serif; color: #12263A; }
.ficha p { margin: 0; font-size: .92rem; color: #3B4F63; }
.ficha.activa { border: 2px solid #0F8B8D; }
.ok { color: #0F8B8D; font-weight: 600; } .mal { color: #B3261E; font-weight: 600; }
</style>
"""
