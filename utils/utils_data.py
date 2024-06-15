import random
from typing import Tuple

import gmpy2
from FAdo.conversions import NFA, GFA

def connect(n: int, k: int, nfa: gmpy2.mpz) -> gmpy2.mpz:
    b = n - 1
    visited = [0] * n
    unvisited = [i + 1 for i in range(b)]
    for a in range(1, n):
        foo = random.randint(0, a - 1)
        bar = random.randint(0, b - 1)
        src = visited[foo]
        dst = unvisited[bar]
        lbl = random.randint(0, k - 1)
        nfa = nfa.bit_set(src * n * k + dst * k + lbl)
        visited[a] = dst
        a += 1
        b -= 1
        del unvisited[bar]
    return nfa

def add_random_transitions(nfa: gmpy2.mpz, size: int, t: int) -> gmpy2.mpz:
    if not t:
        return nfa
    unused = []
    for i in range(size):
        if not nfa.bit_test(i):
            unused.append(i)
    j = len(unused)
    for i in range(t):
        foo = random.randint(0, j - 1)
        nfa = nfa.bit_set(unused[foo])
        del unused[foo]
        j -= 1
    return nfa

def generate(n: int, k: int, d: float) -> Tuple[gmpy2.mpz, gmpy2.mpz]:
    size = n * n * k
    nfa = gmpy2.mpz()
    finals = gmpy2.mpz()
    if d < 0:
        d = random.random()
    nfa = connect(n, k, nfa)
    t = int(d * n * n * k - (n - 1))
    if t < 0:
        print(f"error: it is not possible to have an accessible NFA with {n} states and a transition density as low as {d}")
        exit()
    else:
        nfa = add_random_transitions(nfa, size, t)
    rstate = gmpy2.random_state(random.randint(0, 2147483647 - 1))
    finals = gmpy2.mpz_rrandomb(rstate, n)
    return nfa, finals

def shuffle_fa(fa: NFA) -> None:
    # order[current index] = new index
    order = {}
    lst = [i for i in range(2, len(fa.States))]
    random.shuffle(lst)
    for i in range(len(fa.States)):
        if i == get_initial_state_index(fa):
            order[i] = 0
        elif i == get_final_state_index(fa):
            order[i] = 1
        else:
            order[i] = lst.pop()
    fa.reorder(order)

def get_initial_state_index(fa: NFA) -> int:
    # Implementation choice: https://stackoverflow.com/questions/59825/how-to-retrieve-an-element-from-a-set-without-removing-it
    for state in fa.Initial:
        return state

def get_final_state_index(fa: NFA) -> int:
    # Implementation choice: https://stackoverflow.com/questions/59825/how-to-retrieve-an-element-from-a-set-without-removing-it
    for state in fa.Final:
        return state

def rename_states(fa: NFA, max_n: int) -> None:
    '''
    0: initial state
    max_n + 1: final state
    '''
    lst = [str(i) for i in range(1, max_n + 1)]
    sampled_states_number = random.sample(lst, len(fa.States) - 2)
    names = [None] * len(fa.States)
    for i in range(len(names)):
        if i == get_initial_state_index(fa):
            names[i] = "0"
        elif i == get_final_state_index(fa):
            names[i] = f"{max_n + 1}"
        else:
            names[i] = sampled_states_number.pop()
    garbage_name = ['init' for _ in range(len(fa.States))]
    fa.renameStates(garbage_name)
    fa.renameStates(names)

def convert_nfa_to_gfa(nfa: NFA) -> GFA:
    #Counterpart of FA2GFA function
    gfa = GFA()
    gfa.setSigma(nfa.Sigma)
    gfa.Initial = get_initial_state_index(nfa)
    gfa.States = nfa.States[:]
    gfa.setFinal(nfa.Final)
    gfa.predecessors = {}
    for i in range(len(gfa.States)):
        gfa.predecessors[i] = set([])
    for s in nfa.delta:
        for c in sorted(nfa.delta[s].keys(), reverse=False):
            for s1 in nfa.delta[s][c]:
                gfa.addTransition(s, c, s1)
    for i in range(len(gfa.States)):
        if i not in gfa.delta:
            gfa.delta[i] = {}
    return gfa

def get_random_nfa(n: int, k: int, d: float, max_n: int) -> NFA:
    nfa, finals = generate(n, k, d)
    fa = NFA()
    for i in range(n + 2):
        fa.addState()
    fa.addTransition(0, '@epsilon', 1)
    size = n * n * k
    for i in range(n):
        if finals.bit_test(i):
            fa.addTransition(i + 1, '@epsilon', n + 1)
    for i in range(size):
        if nfa.bit_test(i):
            src = i // (n * k)
            foo = i % (n * k)
            dst = foo // k
            lbl = foo % k
            fa.addTransition(src + 1, str(lbl), dst + 1)
    fa.setInitial({0})
    fa.setFinal({n + 1})
    fa = fa.trim()
    fa = fa.lrEquivNFA()
    shuffle_fa(fa)
    rename_states(fa, max_n)
    return convert_nfa_to_gfa(fa)

if __name__ == "__main__":
    nfa = get_random_nfa(5, 2, 0.1, 10)
    #nfa.display("nfa")
    print(nfa.States)
    print(nfa.Initial)
    print(nfa.Final)
