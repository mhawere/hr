#!/usr/bin/env python3
"""
Script to add visualization features to ai.py and update base.html for heat map display
Usage: python add_visualization_features.py
"""

import re
import os
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create a backup of the file"""
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backup created: {backup_path}")
    return backup_path

# Create visualization utility file content
VISUALIZATION_UTILS_CONTENT = '''"""
Visualization utilities for HR Intelligence System
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO
import base64
import numpy as np
from typing import Dict, Any, List, Tuple

def generate_heat_map_image(heat_map_data: Dict[str, Any], title: str = "Heat Map", 
                           color_scale: Dict[str, Any] = None) -> str:
    """Generate a heat map visualization and return as base64 encoded image"""
    
    try:
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Extract data
        if not heat_map_data:
            return generate_error_image("No data available for visualization")
        
        departments = list(heat_map_data.keys())
        values = [item.get('value', 0) for item in heat_map_data.values()]
        
        # Create figure with better size
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Determine color map based on values
        if color_scale and 'gradient' in color_scale:
            # Use custom gradient
            cmap = plt.cm.RdYlGn
        else:
            cmap = plt.cm.RdYlGn
        
        # Normalize values for color mapping
        norm = plt.Normalize(vmin=min(values) if values else 0, 
                           vmax=max(values) if values else 100)
        colors = cmap(norm(values))
        
        # Create bar chart with gradient colors
        bars = ax.bar(range(len(departments)), values, color=colors, edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        ax.set_xticks(range(len(departments)))
        ax.set_xticklabels(departments, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Value (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add grid
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Set y-axis limits
        ax.set_ylim(0, max(values) * 1.1 if values else 100)
        
        # Add color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Performance Scale', rotation=270, labelpad=20, fontsize=10)
        
        # Add statistics box
        if values:
            stats_text = f'Avg: {np.mean(values):.1f}%\\nMax: {max(values):.1f}%\\nMin: {min(values):.1f}%'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Tight layout
        plt.tight_layout()
        
        # Save to BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        plt.close()
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer.read()).decode()
        return image_base64
        
    except Exception as e:
        plt.close()
        return generate_error_image(f"Error generating visualization: {str(e)}")

def generate_comparison_chart(data1: Dict[str, Any], data2: Dict[str, Any], 
                            labels: Tuple[str, str], title: str = "Comparison Chart") -> str:
    """Generate a comparison chart for two datasets"""
    
    try:
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Prepare data
        categories = list(set(data1.keys()) | set(data2.keys()))
        values1 = [data1.get(cat, {}).get('value', 0) if isinstance(data1.get(cat, {}), dict) else data1.get(cat, 0) for cat in categories]
        values2 = [data2.get(cat, {}).get('value', 0) if isinstance(data2.get(cat, {}), dict) else data2.get(cat, 0) for cat in categories]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Bar positions
        x = np.arange(len(categories))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, values1, width, label=labels[0], color='#667eea', alpha=0.8)
        bars2 = ax.bar(x + width/2, values2, width, label=labels[1], color='#764ba2', alpha=0.8)
        
        # Customize
        ax.set_xlabel('Categories', fontsize=12, fontweight='bold')
        ax.set_ylabel('Values', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save to BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        plt.close()
        
        # Convert to base64
        return base64.b64encode(buffer.read()).decode()
        
    except Exception as e:
        plt.close()
        return generate_error_image(f"Error generating comparison: {str(e)}")

def generate_error_image(error_message: str) -> str:
    """Generate an error image when visualization fails"""
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, f'⚠️ {error_message}', transform=ax.transAxes,
           fontsize=12, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    plt.close()
    
    return base64.b64encode(buffer.read()).decode()
'''

# Updates for ai.py
AI_PY_IMPORTS_UPDATE = '''from fastapi.responses import StreamingResponse
import base64
from io import BytesIO'''

AI_PY_VISUALIZATION_UPDATE = '''
# Add this to the _create_heat_maps function after generating heat_map_data
                
                # Generate visualization if data exists
                if heat_map_data:
                    try:
                        from utils.visualizations import generate_heat_map_image
                        
                        # Create title based on metric and grouping
                        viz_title = f"{metric.replace('_', ' ').title()} by {group_by.replace('_', ' ').title()}"
                        if time_period:
                            viz_title += f" ({time_period.replace('_', ' ')})"
                        
                        # Generate the image
                        image_base64 = generate_heat_map_image(
                            heat_map_data, 
                            title=viz_title,
                            color_scale=result.get("color_scale")
                        )
                        
                        result["visualization"] = {
                            "type": "image/png",
                            "encoding": "base64",
                            "data": image_base64,
                            "filename": f"heatmap_{metric}_{group_by}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        }
                        
                        result["download_instructions"] = [
                            "📊 Heat map visualization generated successfully",
                            "💾 You can download the image directly from the chat",
                            "📈 The visualization shows " + viz_title.lower()
                        ]
                        
                    except Exception as viz_error:
                        logger.warning(f"Visualization generation failed: {viz_error}")
                        result["visualization_error"] = str(viz_error)'''

AI_PY_DOWNLOAD_ENDPOINT = '''
@ai.post("/download-visualization")
async def download_visualization(
    request: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate and download visualization"""
    try:
        viz_data = request.get("visualization", {})
        
        if not viz_data or "data" not in viz_data:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="No visualization data provided"
            )
        
        # Decode base64 image
        image_bytes = base64.b64decode(viz_data["data"])
        filename = viz_data.get("filename", f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        return StreamingResponse(
            BytesIO(image_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"Download visualization error: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download visualization: {str(e)}"
        )'''

# Updates for base.html
BASE_HTML_VISUALIZATION_JS = '''
            // Enhanced message handling for visualizations
            function addMessage(content, isUser = false, isLoading = false, data = null) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `tessa-message ${isUser ? 'user' : 'assistant'}${isLoading ? ' loading' : ''}`;
                
                const avatar = document.createElement('div');
                avatar.className = 'tessa-message-avatar';
                avatar.innerHTML = isUser ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
                
                const messageContent = document.createElement('div');
                messageContent.className = 'tessa-message-content';
                
                if (isLoading) {
                    messageContent.innerHTML = `
                        <span>Tessa is thinking...</span>
                        <div class="tessa-typing-indicator">
                            <div class="tessa-typing-dot"></div>
                            <div class="tessa-typing-dot"></div>
                            <div class="tessa-typing-dot"></div>
                        </div>
                    `;
                } else {
                    // Parse content for special formatting
                    let formattedContent = content;
                    
                    // Handle visualization data
                    if (data && data.visualization && data.visualization.type === 'image/png') {
                        formattedContent += createVisualizationHTML(data.visualization);
                    }
                    
                    messageContent.innerHTML = formattedContent;
                }
                
                messageDiv.appendChild(avatar);
                messageDiv.appendChild(messageContent);
                messagesContainer.appendChild(messageDiv);
                scrollToBottom();
                
                return messageDiv;
            }
            
            // Create visualization HTML
            function createVisualizationHTML(vizData) {
                const imageId = 'viz_' + Date.now();
                const html = `
                    <div class="tessa-visualization" style="margin-top: 15px;">
                        <img id="${imageId}" 
                             src="data:${vizData.type};base64,${vizData.data}" 
                             alt="Data Visualization" 
                             style="width: 100%; max-width: 500px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); cursor: pointer;"
                             onclick="window.open(this.src, '_blank')">
                        <div style="margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap;">
                            <button onclick="downloadVisualization('${imageId}', '${vizData.filename || 'visualization.png'}')" 
                                    style="background: #667eea; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 12px;">
                                <i class="fas fa-download"></i> Download
                            </button>
                            <button onclick="window.open(document.getElementById('${imageId}').src, '_blank')" 
                                    style="background: #764ba2; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 12px;">
                                <i class="fas fa-expand"></i> View Full Size
                            </button>
                        </div>
                    </div>
                `;
                return html;
            }
            
            // Download visualization function
            function downloadVisualization(imageId, filename) {
                const img = document.getElementById(imageId);
                if (!img) return;
                
                const link = document.createElement('a');
                link.href = img.src;
                link.download = filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }'''

def update_ai_py(filepath):
    """Update ai.py with visualization features"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if updates already exist
    if "generate_heat_map_image" in content:
        print("⚠️ Visualization features already exist in ai.py")
        return False
    
    # Add imports if not present
    if "from fastapi.responses import StreamingResponse" not in content:
        import_pattern = r'(from fastapi import.*?)\n'
        import_match = re.search(import_pattern, content)
        if import_match:
            insert_pos = import_match.end()
            content = content[:insert_pos] + AI_PY_IMPORTS_UPDATE + "\n" + content[insert_pos:]
    
    # Find _create_heat_maps function and update it
    heat_map_pattern = r'(result\["heat_map_data"\] = heat_map_data)(.*?)(return result)'
    heat_map_match = re.search(heat_map_pattern, content, re.DOTALL)
    
    if heat_map_match:
        insert_pos = heat_map_match.end(1)
        content = content[:insert_pos] + "\n" + AI_PY_VISUALIZATION_UPDATE + content[insert_pos:]
        print("✅ Updated _create_heat_maps function with visualization")
    else:
        print("⚠️ Could not find _create_heat_maps function pattern")
    
    # Add download endpoint before the health check endpoint
    health_pattern = r'(@ai\.get\("/health"\))'
    health_match = re.search(health_pattern, content)
    
    if health_match:
        insert_pos = health_match.start()
        content = content[:insert_pos] + AI_PY_DOWNLOAD_ENDPOINT + "\n\n" + content[insert_pos:]
        print("✅ Added download-visualization endpoint")
    else:
        print("⚠️ Could not find health endpoint to insert download endpoint")
    
    # Write updated content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True

def update_base_html(filepath):
    """Update base.html with visualization handling"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if visualization code already exists
    if "createVisualizationHTML" in content:
        print("⚠️ Visualization features already exist in base.html")
        return False
    
    # Find the addMessage function and replace it with enhanced version
    add_message_pattern = r'(// Add message to chat\s*function addMessage\(.*?\{.*?\})'
    add_message_match = re.search(add_message_pattern, content, re.DOTALL)
    
    if add_message_match:
        # Find the end of sendMessage function to insert our enhanced code
        send_message_pattern = r'(async function sendMessage\(\) \{.*?// Add Tessa\'s response\s*)(addMessage\(data\.response.*?\);)'
        send_message_match = re.search(send_message_pattern, content, re.DOTALL)
        
        if send_message_match:
            # Replace the addMessage call with enhanced version
            old_call = send_message_match.group(2)
            new_call = "addMessage(data.response || 'Sorry, I encountered an error processing your request.', false, false, data);"
            content = content.replace(old_call, new_call)
            print("✅ Updated sendMessage to pass data to addMessage")
        
        # Now insert the enhanced functions before initializeTessaChat closing brace
        init_tessa_pattern = r'(function initializeTessaChat\(\) \{.*?)(// Event listeners)'
        init_tessa_match = re.search(init_tessa_pattern, content, re.DOTALL)
        
        if init_tessa_match:
            insert_pos = init_tessa_match.start(2)
            content = content[:insert_pos] + BASE_HTML_VISUALIZATION_JS + "\n\n            " + content[insert_pos:]
            print("✅ Added visualization handling functions")
    else:
        print("⚠️ Could not find addMessage function pattern")
    
    # Write updated content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True

def create_visualization_utils():
    """Create the visualization utilities file"""
    utils_dir = "utils"
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
        print(f"✅ Created {utils_dir} directory")
    
    viz_file = os.path.join(utils_dir, "visualizations.py")
    
    if os.path.exists(viz_file):
        print(f"⚠️ {viz_file} already exists")
        return False
    
    with open(viz_file, 'w', encoding='utf-8') as f:
        f.write(VISUALIZATION_UTILS_CONTENT)
    
    print(f"✅ Created {viz_file}")
    
    # Create __init__.py if it doesn't exist
    init_file = os.path.join(utils_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("")
        print(f"✅ Created {init_file}")
    
    return True

def update_requirements():
    """Update requirements.txt with visualization dependencies"""
    req_file = "requirements.txt"
    
    new_deps = [
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "pandas>=1.3.0"
    ]
    
    if os.path.exists(req_file):
        with open(req_file, 'r') as f:
            existing = f.read()
        
        deps_to_add = []
        for dep in new_deps:
            dep_name = dep.split('>=')[0]
            if dep_name not in existing:
                deps_to_add.append(dep)
        
        if deps_to_add:
            with open(req_file, 'a') as f:
                f.write("\n# Visualization dependencies\n")
                for dep in deps_to_add:
                    f.write(f"{dep}\n")
            print(f"✅ Added visualization dependencies to {req_file}")
        else:
            print(f"⚠️ Visualization dependencies already in {req_file}")
    else:
        print(f"⚠️ {req_file} not found")

# Main execution
if __name__ == "__main__":
    print("🚀 Adding Visualization Features to HR Intelligence System\n")
    
    # File paths
    ai_file = "routers/ai.py"
    base_file = "templates/base.html"
    
    # Check if files exist
    if not os.path.exists(ai_file):
        print(f"❌ {ai_file} not found!")
        exit(1)
    
    if not os.path.exists(base_file):
        print(f"❌ {base_file} not found!")
        exit(1)
    
    # Create backups
    ai_backup = backup_file(ai_file)
    base_backup = backup_file(base_file)
    
    try:
        # Create visualization utilities
        create_visualization_utils()
        
        # Update ai.py
        print("\n📝 Updating ai.py...")
        update_ai_py(ai_file)
        
        # Update base.html
        print("\n📝 Updating base.html...")
        update_base_html(base_file)
        
        # Update requirements
        print("\n📝 Updating requirements...")
        update_requirements()
        
        print("\n✨ Visualization features successfully added!")
        print("\n📊 You can now:")
        print("  - Ask for heat maps and see them as images")
        print("  - Download visualizations directly from chat")
        print("  - View full-size images in new tabs")
        print("\n🎯 Example queries:")
        print("  - 'Show me an attendance heat map by department'")
        print("  - 'Create a heat map of punctuality by day of week'")
        print("  - 'Generate attendance heat map for last 30 days'")
        
        print("\n⚡ Don't forget to install dependencies:")
        print("  pip install matplotlib seaborn pandas")
        
    except Exception as e:
        print(f"\n❌ Error during update: {str(e)}")
        print(f"Backups available at:")
        print(f"  - {ai_backup}")
        print(f"  - {base_backup}")