class Ontonotes5Features:
    ontonotes_labels = [x.strip().replace(',', '') for x in
                        """CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART""".split()]
    ontonotes_labels_with_o = ['O'] + [x.strip().replace(',', '') for x in
                                       """CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART""".split()]
    ontonotes_labels_bio = ['O'] + [f'B-{x}' for x in ontonotes_labels] + [f'I-{x}' for x in ontonotes_labels]
    id_to_label = dict(enumerate(ontonotes_labels_bio))
    label_to_id = {v: k for k, v in id_to_label.items()}


def get_non_empty_ner(tags):
    for t in tags:
        if t != 'O':
            return t
    return 'O'


def get_iob(t):
    if t.ent_iob_ == 'O':
        return 'O'
    return t.ent_iob_ + "-" + t.ent_type_