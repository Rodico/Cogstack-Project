import base64
import zlib
import requests

# Your chart data formatted for the renderer
mermaid_code = """
graph TD
    classDef green fill:#E8F5E9,stroke:#81C784,stroke-width:1px,color:#1B5E20;
    classDef orange fill:#FFF3E0,stroke:#FFB74D,stroke-width:1px,color:#E65100;
    classDef red fill:#FFEBEE,stroke:#E57373,stroke-width:1px,color:#B71C1C;
    classDef grey fill:#F5F5F5,stroke:#BDBDBD,stroke-width:1px,color:#424242;

    A["<b>smoking_findings_only pack</b><br><font size=2>MedCAT v2 format (cdb/ folder)</font>"]:::green
    B["<b>Uploaded to MedCATtrainer</b><br><font size=2>Trainer runs MedCAT v1.16</font>"]:::orange
    C["<b>Format mismatch</b><br><font size=2>v1 expects flat cdb.dat, not a folder</font>"]:::red
    D["<b>CDB extracts empty</b><br><font size=2>~4KB stub vs ~255MB real data</font>"]:::red
    E["<b>No concepts imported</b><br><font size=2>Nothing highlights, search fails</font>"]:::grey

    A --> B
    B --> C
    C --> D
    D --> E

    Title["<b>Attempted fixes — all blocked</b>"]:::grey
    E --> Title

    F["<b>Convert v2 → v1</b><br><font size=2>No converter exists;<br>v2 only writes folders</font>"]:::grey
    G["<b>Rebuild Trainer on v2</b><br><font size=2>Image builds, but app<br>won't start on v2 API</font>"]:::grey
    
    Title --> F
    Title --> G

    H["<b>Workaround used</b><br><font size=2>Ran pack via MedCAT v2 library; Trainer demo on v1 model</font>"]:::green

    F --> H
    G --> H

    Footer["Root cause: Trainer (v1) not yet compatible with v2 model packs"]:::grey
    H --> Footer
    
    style Title stroke-width:0px,fill:none
    style Footer stroke-width:0px,fill:none
"""

# Compress and encode the diagram for the API request
payload = base64.urlsafe_b64encode(zlib.compress(mermaid_code.encode('utf-8'), 9)).decode('ascii')
url = f'https://kroki.io/mermaid/png/{payload}'

try:
    # Fetch the generated PNG file from the open-source API
    response = requests.get(url)
    if response.status_code == 200:
        # Save it directly right next to your script
        with open("medcat_mismatch_flowchart.png", "wb") as f:
            f.write(response.content)
        print("Success! 'medcat_mismatch_flowchart.png' has been saved directly to your current folder.")
    else:
        print(f"Failed to generate image. Server returned status code: {response.status_code}")
except Exception as e:
    print(f"An error occurred: {e}. Make sure you are connected to the internet.")