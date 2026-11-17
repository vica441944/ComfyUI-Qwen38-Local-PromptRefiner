from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .beat_pack import register_routes
from .director_pack_writeback import register_routes as register_writeback_routes
from .director_face_refine_mem import register_patch

register_patch()
register_routes()
register_writeback_routes()

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
