import carla
import time

client = carla.Client('localhost', 2000)
client.set_timeout(20.0)

# print(client)

# world = client.get_world()

# print(world)

print(client.get_available_maps())

# world = client.load_world('ait_v4')
# print('load world done.')

# settings = world.get_settings()

# print(settings)
