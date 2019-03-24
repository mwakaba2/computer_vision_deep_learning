## Question 5: Research Paper

Please review this paper which won the best paper for 2018 at CVPR: [https://arxiv.org/pdf/1804.08328.pdf](https://arxiv.org/pdf/1804.08328.pdf) Answer the following questions:

1. What are the key findings for this paper?
2. What ground-breaking applications can you think of based on this paper's findings?
3. The paper only showed a first-order task affinity matrix for all of the tasks, assuming we would like to extend the findings of the paper and try to compute a 2-D affinity matrix for all of the tasks, how would you go about designing a procedure for developing such a flow?


### Answer

1.
	At a high level, the paper proposes a computational approach to creating a taxonomic map of different vision tasks and their relationships. For a company like Polarr and its technology, this means new vision tasks can be solved more quickly with limited datasets and resources.

	In more technical terms, the researcher's resulting map has knowledge of obscure relationships between vision tasks that humans cannot normally decipher. The map can output what combination of tasks a target task would benefit from transfer learning. Currently, most transfer learning tasks are limited to using the imagenet model and relying on human intuition, but this map can produce an adaptive and efficient transfer policy for any given target task based on performance. With transfer learning, we can avoid learning from scratch. This helps reduce training time and data needed to solve new problems without compromising performance (compared to individually trained tasks).

2.
	With this map, we can solve a set of similar and related tasks instead of solving each individually. Here are several applications that I think can benefit from this map.

	* Photo improvements: Adding color to a digitized black and white photo. Colorization can be transfer learned using second-order transfers of `(3D Keypoints, Object recognition)`. If a photo was taken in a room, the knowledge from the 3D Keypoints will help us understand the room layout. This will help identify objects in the right context and add the most appropriate color.

	* Photo Categorization: Since the training data used in the research paper are all indoor scenes, one immediate use-case is categorizing photos by different types of rooms (e.g. living room, office, kitchen, dining room). For example, I can see this being potentially useful for companies like Zillow and Houzz that categorize and extract insights from their massive interior photo collections. This can be done by creating an interior classification model using second-order transfers of `(3D Keypoints, Object Recognition)`.

	* Privacy: We can blur people's faces, copyrighted images/paintings in picture frames, and sensitive documents depending on the scene in the video. For example, if we're in a office, we can blur out any papers on the desk. This will require a scene detection model using third-order transfers of `(3D Keypoints, Curvature, Object Recognition)`.


3.
	Assuming the target task's first-order affinity matrix stores the source task s<sub>i</sub>'s performance values (in comparison with all other source tasks' performance values), the 2-D affinity matrix would be the second-order transfers' affinity matrix. The goal of the target task's second-order affinity matrix is to store the pair of source tasks' (s<sub>1</sub>, s<sub>2</sub>) performance values (in comparison with other pairs).

	For second-order transfers, there can be |T| x <sub>|S|</sub>C<sub>2</sub> possible combinations of transfers. Training all possible combinations will be computationally expensive, so I would first filter out low performant pairs. I would look at the the target t's first order performances to pick the top 5 source tasks. Next, I will create 10 pairs with the 5 source tasks to train the transfer functions. The second order matrix's calculation will be similar to the first order transfers' affinity matrix P. However, for each t, I'll be creating a pairwise tournament matrix between the 10 pairs.
