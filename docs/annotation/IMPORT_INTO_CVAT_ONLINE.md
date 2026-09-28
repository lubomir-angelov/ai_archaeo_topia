Importing into CVAT online

Files to upload (seen from Windows):
- Labels: \\wsl$\Ubuntu\home\ubuntu\repos\ai_archaeo_topia\annotation\cvat\labels_cvat_raw.json. This is v0.0.4, with the labels mound (polygon), hard_negative_symbol (rectangle) and uncertain_ignore (rectangle).
- Images: C:\Users\lubom\ai_archaeo_topia\data_lake\cleaned\map_clips\_frozen\<sheet>\<sheet>_1..4.png. That's 16 RGBA PNGs of about 2.4k×2.2k pixels each, which CVAT accepts as they are.

1. Create the project and import the labels

1. Go to app.cvat.ai, open Projects, click +, then Create a new project.
2. Give it a name, e.g. mounds_blind_test_frozen. Only these 4 sheets should ever go into this project.
3. In the Labels section, open the Raw tab. Delete the empty [], paste the whole contents of labels_cvat_raw.json and click Done.
4. Check that there are 3 labels. mound should show 15 attributes and hard_negative_symbol 16, including negative_type.
5. Click Submit & Open.

2. Create one task per sheet (repeat 4 times)

1. Inside the project, click + and then Create a new task.
2. Name it after the sheet, e.g. task_K-35-22-A-v. The other three are task_K-35-39-G-v, task_K-35-39-V-g and task_L-35-139-V-v.
3. Leave Project set to the one you just made; the task takes its labels from it.
4. Set Subset to Test.
5. Under Select files, choose My computer and drag in only that sheet's 4 PNGs. Don't upload clips.json.
6. Open Advanced configuration and set:
   - Image quality: 100. The default of 70 compresses the images, which blurs the small mound symbols and contour lines.
   - Sorting method: Lexicographical, so the pieces stay in _1.._4 order.
   - Leave Segment size empty, giving one job of 4 images. If two people will share a sheet, set it to 2.
7. Click Submit & Open, and wait until the task status says the data is ready.

Keep the file names exactly as they are. clips.json links each file name to its position and coordinates on the map, so a renamed file can no longer be placed on the map.

3. Rules for annotating the blind set

- Don't upload any annotations or model predictions into these tasks.
- Don't use CVAT's AI Tools, Automatic annotation or the SAM interactor either. The protocol says the annotator must see no model output. If you want to allow them, decide that before anyone starts, and write it down.
- The attribute defaults already fit blind annotation: annotation_provenance=human_added, review_status=unreviewed, detector_confidence empty.
- Follow docs/annotation/PROTOCOL_EN.md, and assign each job under Jobs → Assignee.

4. Exporting later

When annotation is finished, export from the project: Actions → Export dataset → COCO 1.0, with images off. Keep that export separate from the training exports in annotation/cvat/v0.0.x. Anything that goes into a training pipeline must never include it.
