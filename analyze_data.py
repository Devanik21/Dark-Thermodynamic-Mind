import zipfile
import json
import os
import numpy as np

zip_path = r"C:\Users\debna\Downloads\causa-sui-main\causa-sui-main\Streamlit App\Zero_Point_Genesis\genesis_data.zip"

if not os.path.exists(zip_path):
    print(f"Error: File not found at {zip_path}")
    exit()

try:
    with zipfile.ZipFile(zip_path, 'r') as z:
        print("Files in zip:", z.namelist())
        
        # 1. Analyze Stats
        if 'stats.json' in z.namelist():
            with z.open('stats.json') as f:
                stats = json.load(f)
                if stats:
                    df = pd.DataFrame(stats) if 'pd' in locals() else stats
                    print(f"\n--- Stats Analysis ({len(stats)} ticks) ---")
                    print(f"Final Population: {stats[-1]['population']}")
                    print(f"Final Avg Energy: {stats[-1]['avg_energy']}")
                    
                    # Calculate Survival Rate Analysis
                    populations = [s['population'] for s in stats]
                    thoughts = [s['thoughts'] for s in stats]
                    print(f"Max Population: {max(populations)}")
                    print(f"Min Population: {min(populations)}")
                    print(f"Total Thoughts (Learning Events): {sum(thoughts)}")
        
        # 2. Analyze Gene Pool
        if 'genes.json' in z.namelist():
            with z.open('genes.json') as f:
                genes = json.load(f)
                print(f"\n--- Gene Pool Analysis ---")
                print(f"Pool Size: {len(genes)}")
                if len(genes) > 0:
                    # Check for convergence (variance of weights)
                    # We'll just look at the first layer weight mean of first gene
                    sample_layer = list(genes[0].keys())[0]
                    print(f"Sample Gene Layer: {sample_layer}")
        
        # 3. Analyze Events
        if 'events.json' in z.namelist():
            with z.open('events.json') as f:
                events = json.load(f)
                print(f"\n--- Event Log Analysis ({len(events)} events) ---")
                # Count event types
                deaths = sum(1 for e in events if "DIED" in e['Event'])
                births = sum(1 for e in events if "BORN" in e['Event'])
                rewires = sum(1 for e in events if "REWIRING" in e['Event'])
                print(f"Deaths: {deaths}")
                print(f"Births: {births}")
                print(f"Rewirings: {rewires}")
                
except Exception as e:
    print(f"Error reading zip: {e}")
