import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO
import base64

def generate_heat_map_image(heat_map_data: dict, title: str = "Attendance Heat Map") -> str:
    """Generate a downloadable heat map image"""
    
    # Convert data to DataFrame
    departments = list(heat_map_data.keys())
    values = [item['value'] for item in heat_map_data.values()]
    
    df = pd.DataFrame({
        'Department': departments,
        'Attendance Rate': values
    })
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create bar chart styled as heat map
    colors = plt.cm.RdYlGn(df['Attendance Rate'] / 100)
    bars = plt.bar(df['Department'], df['Attendance Rate'], color=colors)
    
    # Styling
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Department', fontsize=12)
    plt.ylabel('Attendance Rate (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Add color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Attendance Rate', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    # Save to BytesIO
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    # Convert to base64
    image_base64 = base64.b64encode(buffer.read()).decode()
    return image_base64