from pyctlib import Logger
from pyctlib.logging import upload_file_to_row_property

logger = Logger(True, notion_page_link="https://www.notion.so/zhangyiteng/61a0597e1343442380da3ec05cd93676")

logger.update_notion("training_process", "./trajectory_2000.pdf", True)
logger.update_notion("grid_cells", "1")

upload_file_to_row_property(logger._notion_client, logger._notion_page, "./trajectory_2000.pdf", "training_process")
