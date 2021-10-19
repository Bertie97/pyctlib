from zytlib import Logger
from zytlib.logging import upload_file_to_row_property
import random

logger = Logger(True, True, notion_page_link="https://www.notion.so/zhangyiteng/61a0597e1343442380da3ec05cd93676")

for step in range(20):

    logger.variable("train.loss", step ** 0.5)
    logger.variable("val.loss", step ** 0.5)
    logger.variable("loss[train]", step ** 0.5 + random.random())
    logger.variable("loss[val]", step ** 0.5 + 0.5 * random.random())
    logger.variable("train.x", step ** 0.5)
    logger.variable("train.y", step ** 0.5)
    logger.variable("train.z", step ** 0.5)
    logger.variable("train.w[a]", 1 + random.random())
    logger.variable("train.w[b]", 2 + random.random())
    logger.variable("train.w[c]", 3 + random.random())
    logger.variable("train.w[d]", 4 + random.random())

logger.upload_variable_dict_to_notion("training_process")

# for index in range(100):
#     log

# logger.update_notion("training_process", "./trajectory_2000.pdf", True)
logger.update_notion("grid_cells", "1")

# upload_file_to_row_property(logger._notion_client, logger._notion_page, "./trajectory_2000.pdf", "training_process")
