sture = []
for n in ['n', 'PER', 'LOC', 'ORG']:
    for adv in ['', 'd']:
        for v in ['v', 'vd', 'vn']:
            for adj in ['', 'a', 'ad', 'q', 'm']:
                for z in ['n', 'PER', 'LOC', 'ORG']:
                    sture.append(n+adv+v+adj+z)
print(sture)