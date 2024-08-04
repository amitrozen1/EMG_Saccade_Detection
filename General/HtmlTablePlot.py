from dataclasses import dataclass
from enum import Enum
import base64
import os

class ColType(Enum):
    Text = 0
    Photo = 1
    PhotoSrc = 2

def toHtmlColumn(s):
    return f"<td align='center'>{s}</td>"

def toHtmlPhotoColumn(b64Photo):
    return toHtmlColumn(f'<img src="data:image/png;base64, {b64Photo}" style="display: block; max-width:400px; max-height:400px; width: auto; height: auto;" />')

@dataclass
class HtmlCol:
    Type : ColType
    Value : str

    def toColumn(self) -> str :
        if ( self.Type == ColType.Text ):
            return toHtmlColumn(self.Value)
        elif ( self.Type == ColType.Photo ):
            return toHtmlPhotoColumn(self.Value)
        elif ( self.Type == ColType.PhotoSrc ):
            return toHtmlColumn(f'<img src="{self.Value}" style="display: block; max-width:400px; max-height:400px; width: auto; height: auto;" />')


def initHtmlDictionary(titles):
    d = dict()
    d["titles"] = ["row_id"] + titles
    return d

def addRow(d : dict, row : list):
    if ( not "rows" in d ):
        d["rows"] = list()
    new_id = len(d["rows"])
    d["rows"] += [[HtmlCol(ColType.Text, str(new_id))] + row]

def get_css():
    css = """
    <head>
    <style>
        /* Style the table itself */
        table {
            table-layout: fixed;
            width: 100%;
            border: 1px solid black;
            letter-spacing: 1px;
        }

        /* Make the table header sticky */
        th {
            background-color: #f9f9f9; /* Optional: background color for the header */
            position: sticky;
            top: 0;
            z-index: 1; /* Ensures the header is above other content */
        }

        /* Additional styling for the rest of the table for better visibility */
        td, th {
            border: 1px solid black;
            padding: 8px;
            text-align: left;
        }

        /* Example styling for scrollable content */
        .scrollable-table {
            height: 200px; /* Adjust based on your needs */
            overflow-y: auto;
        }
        </style>
    </head>
    """
    return css

def get_table_tag():
    return "<table letter-spacing='1px'  style='table-layout: fixed; width:100%;' border='1px solid black' >"

def PlotDictionary(dict):
    html = f"<html>{get_css()}{get_table_tag()}"
    #Add titles:
    html += "<tr>"
    for title in dict["titles"]:
        html += f"<th>{title}</th>"
    html += "</tr>"

    #Add rows
    for row in dict["rows"]:
        html+="<tr>"
        for item in row:
            html+= item.toColumn()
        html+="</tr>"
    
    html += "</table></html>"
    return html

def toImageHtmlCol(bytes):
    return HtmlCol(ColType.Photo, base64.b64encode(bytes).decode('ascii'))

def toLinkPhotoHtmlCol(bytes):
    return HtmlCol(ColType.Text, f'<a href="data:image/png;base64, {base64.b64encode(bytes).decode("ascii")}" target="_blank">heatmap</a>')

def toLinkPhotoHtmlColFromPath(p):
    if ( not os.path.exists(p) ):
        return toTextHtmlCol("No such file")
    with open(p, "rb") as f:
        bytes = f.read()
    return toLinkPhotoHtmlCol(bytes)

def toImageHtmlColFromPath(p):
    if ( not os.path.exists(p) ):
        return toTextHtmlCol("No such file")
    with open(p, "rb") as f:
        bytes = f.read()
    return toImageHtmlCol(bytes)


def toImageHtmlColFromPathWithSrc(p):
    return HtmlCol(ColType.PhotoSrc, p)

def toTextHtmlCol(obj):
    return HtmlCol(ColType.Text, str(obj))

def SaveHtmlFile(dic, filename):
    html = PlotDictionary(dic)
    with open(filename, "w") as f:
        f.write(html)

if __name__=="__main__":
    #usage example:
    d = initHtmlDictionary(["i","i*2", "image"])
    photos = [r"C:\Liron\DataEmg\Done\20122022_0830\A.jpeg", r"C:\Liron\Code\LIRON_EMG_DataAnalysis\1.png"]

    for i in range(2):
        addRow(d, [HtmlCol(ColType.Text, str(i)), HtmlCol(ColType.Text, str(i*2)), toImageHtmlColFromPath(photos[i]) ])
    SaveHtmlFile(d, "2.html")
