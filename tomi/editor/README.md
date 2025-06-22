# TOMI Editor GUI (Demo)
We provide a simple graphical interface for visualizing TOMI’s data structure and inspecting the properties of each node.

When you run `tomi_song.gen(open_editor=True)`, a GUI window will open automatically once the composition arrangement is loaded into REAPER:
<div><img src="/img/editor_gui_arrangement.jpg" style="width:95%" alt=""></div>

The `NodeArrangement` panel displays a tabular view of the song arrangement. By default, rows and columns are organized as `Track` × `SongStructure`, similar to a traditional DAW layout. Each cell contains a pair of nodes: a transformation node and a clip node.

Clicking on the `NodeGraph` panel reveals all the composition links in a node-editor-style view:
<div><img src="/img/editor_gui_nodegraph.jpg" style="width:95%" alt=""></div>
It may look a bit cluttered if there are many nodes, lines with the same color belong to the same link.

Clicking the `Song Settings` button brings up the current song settings:
<div><img src="/img/editor_gui_songsettings.jpg" style="width:95%" alt=""></div>

You can modify these settings and click the `Sync Project` button in the upper-right corner to apply them to REAPER. This action will regenerate the song project using the updated settings, including re-running the sample retrieval process and adjusting tempo and key for each clip. Note that this process does not prompt the LLM again. Also, syncing only applies new settings to the project; it does not detect user edits made directly in REAPER or convert them back into TOMI’s data structure.

Clicking any node will display a `Node Settings` panel on the right, where you can view and edit that node’s properties. For example, you can replace the sample used in a clip node and click `Sync Project` to apply the change in REAPER.
<div><img src="/img/editor_gui_nodesettings.jpg" style="width:95%" alt=""></div>