import pandas as pd
import plotly.express as px

d = vipy.util.readjson('/Users/jebyrne/Desktop/cap_label_to_superlabel.json')
df = pd.DataFrame([{'fine':k, 'coarse':v} for (k,v) in d.items()])
fig = px.sunburst(df, path=['coarse', 'fine'], color='coarse', hover_data=['fine'])
fig.write_image("/Users/jebyrne/Desktop/superlabel.pdf")
fig.write_html("/Users/jebyrne/Desktop/superlabel.html")

