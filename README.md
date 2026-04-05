# debateland
growth higgsfield
Narrative-Aware Growth System

This project reverse-engineers Claude’s growth dynamics using multi-platform discourse data (Reddit, YouTube, HackerNews) and proposes a counter-playbook based on narrative control.

We built a system that:
Detects discourse spikes
Analyzes sentiment and narrative themes
Recommends growth or crisis-response actions

Setup
Requirements

Python 3.10+
Libraries:

pandas
numpy
matplotlib
scikit-learn
textblob

Run pipeline

bash
python youtube_scraper.py
python merge_all.py
python quick_analysis.py
python spike_narrative_analysis.py


Outputs

sentiment_over_time.png
spikes_and_narratives.png
engagement_by_platform.png
engagement_by_type.png
narrative_theme_summary.csv

Assumptions

Engagement = likes + comments + upvotes + points
YouTube comments approximate public sentiment (even if noisy)
Keyword tagging is sufficient proxy for narrative themes
Spike detection via robust z-score reflects real-world attention events
Dataset (1,004 records) is directional

Tradeoffs

Used keyword-based topic tagging instead of BERTopic for speed
Limited Reddit/HN sample size due to scraping constraints
Sentiment via TextBlob (simple but fast), not transformer models
No real-time pipeline (offline analysis only)
No full production system for Narrative Risk Engine (conceptual layer)

Key Insight

AI growth is not a funnel — it is a loop:

Trigger → Creator Content → Community Debate → Replication → Distribution

However:

High engagement ≠ trust
Low trust → narrative collapse during crises

Contribution

We introduce:

Narrative-Aware Growth System**

Combines growth strategy + risk management
Intercepts discourse at:

Debate layer (Reddit)
Narrative layer (YouTube)
Spike window (0–72h)

Limitations

Dataset biased toward YouTube (92%)
No real-time monitoring system implemented
No causal inference (correlation-based analysis)

Future Work

Real-time spike detection dashboard
Transformer-based sentiment analysis
Automated response generation (LLM-based)
Integration with social listening APIs

Authors

Diana Shakirova, Merey Bolat, Azhar Zulkharnay
Growth Engineering Track
