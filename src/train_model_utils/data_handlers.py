

def make_batches(list_of_items, batch_number):
    """
    creates N lists of size ``batch_number``

    :param list_of_items: [
                            item1,
                            item2,
                            item3,
                            ...
                          ]
    :param batch_number: int
    :return:
    """
    return [list_of_items[i:i + batch_number] for i in range(0, len(list_of_items), batch_number)]
