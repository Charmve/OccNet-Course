
# %matplotlib inline
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval/', dataroot='/data/nuscenes', verbose=True)

# nusc = NuScenes(version='v1.0-mini/', dataroot='/data/nuscenes/v1.0-mini', verbose=True)

my_scene_token = nusc.field2token('scene', 'name', 'scene-0061')[0]
print(my_scene_token)
# nusc.render_scene_channel(my_scene_token, 'CAM_FRONT')

# my_scene_token = "c3ab8ee2c1a54068a72d7eb4cf22e43d"
nusc.render_scene(my_scene_token)
