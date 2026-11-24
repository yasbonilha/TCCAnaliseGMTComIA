#!/usr/bin/env python3
"""
check_journal.py -- confere as regras de journal TUI que este projeto aprendeu
a duras penas. Rode antes de subir qualquer journal para o cluster.

    ./check_journal.py JOURNAL_solve JOURNAL_post ...

Sai com codigo 1 se houver ERRO. AVISOs nao derrubam o codigo de saida.

REGRAS

 1. /display/contour <campo>  SEMPRE pergunta minimo e maximo, mesmo com
    auto-range? yes -- auto-range apenas pre-preenche os defaults. No nivel
    superior as duas linhas seguintes sao as respostas, e precisam ser:
      - duas linhas em branco  -> aceita o auto-range, ou
      - duas linhas com valor  -> escala fixa
    Uma linha so, ou uma em branco e outra com valor, desalinha o journal: o
    Fluent come a linha seguinte como resposta e TUDO depois anda um lugar.
    Esse prompt desync foi a causa de quase todos os bugs do projeto.

 2. /views/camera/target, /position e /up-vector no nivel superior recebem tres
    valores, um por linha. (A forma de uma linha so vale dentro de strings de
    execute-commands ou de ti-menu-load-string.)

 3. Aspas e parenteses precisam fechar na mesma linha. Comentarios (;) ficam
    de fora dessa conta.

 4. AVISO: (ti-menu-load-string "...") seguido de linha em branco. E a rede de
    seguranca para comando de forma nao verificada -- se o argumento estiver
    errado, a linha em branco responde com o default e o journal segue alinhado.
    Depois que o comando estiver confirmado em log, a rede pode sair.

 5. AVISO: display/set/contours/... antes de haver dados de solucao. Esse menu
    so existe depois de hyb-initialization ou de um /file/read-data. Ler so a
    malha nao basta.

 6. /file/export/ascii <arquivo> tem um bloco de respostas de forma fixa, sem
    linha em branco nenhuma dentro dele:
        ()                <- Surfaces(1) [()]              dominio inteiro
        yes | no          <- Delimiter/Comma? [no]         yes = virgula
        <campo>           <- ASCII scalar(1)>
        ...                  um por linha
        ()                <- FECHA a lista de campos
        yes | no          <- Location/Cell-Centered? [no]  yes = centroide

    A sequencia foi descoberta em log (jobs 5293/5294); a Ansys nao a publica.

    LINHA EM BRANCO NAO FECHA A LISTA DE CAMPOS. O prompt "ASCII scalar(N)>"
    e o unico da sequencia SEM colchetes -- ou seja, sem default -- e responder
    em branco faz o Fluent reimprimir o catalogo inteiro de campos e perguntar
    de novo. No job 5293 isso consumiu 17 linhas, comeu os comentarios do fim
    do journal como se fossem respostas e deixou o /exit orfao.

    REGRA GERAL, que vale para qualquer prompt do Fluent e nao so para este:
        prompt SEM colchetes = sem default = linha em branco NAO responde.

    SUPERFICIE NOMEADA (rodada new_sim_turb, confirmado em sessao interativa
    em 05/10/2026). Em vez do "()" do dominio inteiro, a lista de superficies
    recebe nomes, um por linha, e e FECHADA por "()":
        inlet             <- Surfaces(1) [()]
        ()                <- Surfaces(2) [()]   fecha a lista
        yes               <- Delimiter/Comma?
        ...
    O nome tem que ser uma zona que existe: o ERRO abaixo compara com as
    zonas que o /report/summary listou para esta malha (ZONAS_CONHECIDAS).

 7. /solve/set/limits pergunta, em sequencia e SEM subcomandos (lido em
    sessao interativa, Fluent 25.2, SST sem energia):
        Minimum allowable absolute pressure (in [Pa]) [1]
        Maximum allowable absolute pressure (in [Pa]) [50000000000]
        Minimum allowable k (in [m^2/s^2]) [1e-14]
        Minimum allowable omega (in [s^-1]) [9.999999999999999e-21]
        Maximum allowable turbulent/laminar viscosity ratio [100000]
    Duas formas aceitas no journal:
      - DEFINIR: exatamente 4 linhas "," e depois UM numero (o teto).
        Com 3 ou 5 virgulas o numero cai noutro limite ou no prompt de
        comando -- e nada avisa.
      - ECOAR: so linhas em branco (aceita tudo como esta e imprime os
        valores atuais no log).
    Se ligar energia ou trocar o modelo, a contagem muda: releia em sessao
    interativa e atualize LIMITS_PROMPTS_ANTES_DO_TETO.
"""

import re
import sys

NUM = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
# Zonas desta malha, do /report/summary da sessao interativa de 05/10/2026.
ZONAS_CONHECIDAS = {'inlet', 'outlet', 'lateral_min', 'lateral_max',
                    'ground', 'top', 'cylinder_wall', 'cylinder_top'}
LIMITS_PROMPTS_ANTES_DO_TETO = 4
CAM = ('/views/camera/target', '/views/camera/position', '/views/camera/up-vector')


def check(path):
    try:
        bruto = open(path, 'rb').read()
    except OSError as e:
        print('  ERRO  nao consegui ler: %s' % e)
        return 1, 0

    errors = warnings = 0

    # --- REGRA 0: fim de linha ------------------------------------------------
    # Um journal que passou por Windows vem com CRLF. Isso e MUITO pior aqui do
    # que num arquivo comum: as linhas em branco que respondem aos prompts de
    # /display/contour deixam de ser vazias -- passam a conter um \r -- e o
    # Fluent pode le-las como resposta nao vazia. Resultado: prompt desync, o
    # bug que mais custou neste projeto, com causa invisivel a olho nu.
    # (O mesmo CRLF ja quebrou ranges_from_log.py com "python3\r: not found".)
    if b'\r\n' in bruto:
        n_crlf = bruto.count(b'\r\n')
        print('  ERRO  %s: %d linhas com terminador CRLF (Windows).' % (path, n_crlf))
        print('        As linhas em branco deixam de ser vazias e podem desalinhar')
        print('        os prompts. Corrija com:  sed -i \'s/\\r$//\' %s' % path)
        errors += 1

    lines = bruto.decode('utf-8', errors='replace').replace('\r\n', '\n').split('\n')

    def err(i, msg):
        nonlocal errors
        print('  ERRO  %s:%d  %s' % (path, i + 1, msg))
        errors += 1

    def warn(i, msg):
        nonlocal warnings
        print('  aviso %s:%d  %s' % (path, i + 1, msg))
        warnings += 1

    seen_data = False

    for i, raw in enumerate(lines):
        s = raw.strip()

        if s.startswith('/solve/initialize') or s.startswith('/file/read-data') \
           or s.startswith('/file/read-case-data'):
            seen_data = True

        # --- regra 3: aspas e parenteses, ignorando comentarios ---
        if not s.startswith(';'):
            if raw.count('"') % 2:
                err(i, 'numero impar de aspas: %s' % s[:60])
            if s.startswith('(') and raw.count('(') != raw.count(')'):
                err(i, 'parenteses nao fecham: %s' % s[:60])

        # --- regra 5 ---
        if s.startswith('/display/set/contours/') and not seen_data:
            warn(i, 'display/set/contours antes de dados carregados '
                    '(precisa de hyb-initialization ou read-data antes)')

        # --- regra 1: contour no nivel superior, sem min/max inline ---
        if s.startswith('/display/contour ') and '"' not in raw:
            parts = s.split()
            if len(parts) == 2:                     # forma nua: espera 2 respostas
                nxt = lines[i + 1:i + 3]
                if len(nxt) < 2:
                    err(i, 'contour no fim do arquivo sem as duas linhas de resposta')
                else:
                    a, b = nxt[0].strip(), nxt[1].strip()
                    blank = (a == '' and b == '')
                    valued = (NUM.match(a) is not None and NUM.match(b) is not None)
                    if not (blank or valued):
                        err(i, 'contour precisa de 2 linhas de resposta (2 em branco '
                               'para auto-range, ou 2 valores). Recebi: %r, %r' % (a, b))
            elif len(parts) != 4:
                warn(i, 'contour com %d argumentos; esperava 1 (nu) ou 3 '
                        '(campo min max)' % (len(parts) - 1))

        # --- regra 2: blocos de camera ---
        if s in CAM:
            vals = [l.strip() for l in lines[i + 1:i + 4]]
            if len(vals) < 3 or not all(NUM.match(v) for v in vals):
                err(i, '%s precisa de 3 valores numericos, um por linha. Recebi: %r'
                    % (s, vals))

        # --- regra 6: bloco de /file/export/ascii ---
        if s.startswith('/file/export/ascii ') and len(s.split()) == 2:
            j = i + 1
            YN = ('yes', 'no', 'y', 'n')

            def peek(k):
                return lines[k].strip() if k < len(lines) else None

            surf = peek(j)
            if surf is None or surf == '':
                err(i, 'export/ascii: falta a linha de Surfaces logo abaixo '
                       '(use "()" para o dominio inteiro). Linha em branco '
                       'desalinha o resto do bloco.')
            else:
                # Lista de superficies: "()" sozinho = dominio inteiro; senao
                # nomes, um por linha, ate o "()" que fecha a lista.
                while surf not in (None, '()'):
                    if surf == '' or surf.startswith('/') or surf.startswith(';') \
                       or surf in ('yes', 'no', 'y', 'n'):
                        err(j, 'export/ascii: lista de superficies nao foi fechada '
                               'com "()" antes de %r' % surf[:40])
                        break
                    if surf not in ZONAS_CONHECIDAS:
                        # ERRO, nao aviso: nome invalido deixa o prompt de
                        # superficie pendente e o journal desalinha.
                        err(j, 'export/ascii: superficie %r nao esta entre as zonas '
                               'conhecidas desta malha (%s)'
                             % (surf, ' '.join(sorted(ZONAS_CONHECIDAS))))
                    j += 1
                    surf = peek(j)
                deli = peek(j + 1)
                if deli not in YN:
                    err(i + 1, 'export/ascii: a resposta depois da lista de '
                               'superficies e Delimiter/Comma? e precisa ser yes '
                               'ou no (yes = virgula = CSV). Recebi: %r' % deli)
                # lista de campos ate o "()" que a fecha
                k = j + 2
                fechou = False
                while k < len(lines):
                    v = lines[k].strip()
                    if v == '()':
                        fechou = True
                        break
                    if v == '':
                        err(k, 'export/ascii: LINHA EM BRANCO dentro da lista de '
                               'campos. O prompt "ASCII scalar(N)>" nao tem '
                               'default: em branco o Fluent reimprime o catalogo '
                               'e pergunta de novo, para sempre. Feche com "()".')
                        break
                    if v.startswith('/') or v.startswith(';'):
                        err(k, 'export/ascii: a lista de campos nao foi fechada '
                               'com "()" antes de %r' % v[:40])
                        break
                    k += 1
                if fechou:
                    if k == j + 2:
                        warn(k, 'export/ascii: lista de campos vazia; o arquivo '
                                'sai so com as coordenadas')
                    cc = peek(k + 1)
                    if cc not in YN:
                        err(k, 'export/ascii: depois do "()" que fecha a lista vem '
                               'Location/Cell-Centered?, que precisa ser yes ou no '
                               '(yes = centroide). Recebi: %r' % cc)
                elif k >= len(lines):
                    err(i, 'export/ascii: a lista de campos nunca e fechada com "()"')

        # --- regra 7: /solve/set/limits ---
        if s == '/solve/set/limits':
            k = i + 1
            nxt = lines[k].strip() if k < len(lines) else None
            if nxt == ',':
                c = 0
                while k < len(lines) and lines[k].strip() == ',':
                    c += 1
                    k += 1
                v = lines[k].strip() if k < len(lines) else ''
                if c != LIMITS_PROMPTS_ANTES_DO_TETO:
                    err(i, '/solve/set/limits com %d "," antes do valor; o teto e o '
                           '%do prompt, entao precisam ser exatamente %d. Com outra '
                           'contagem %r cai noutro limite.'
                        % (c, LIMITS_PROMPTS_ANTES_DO_TETO + 1,
                           LIMITS_PROMPTS_ANTES_DO_TETO, v))
                if not NUM.match(v):
                    err(k, '/solve/set/limits: depois das "," vem o teto, um numero. '
                           'Recebi: %r' % v)
                else:
                    seg = lines[k + 1].strip() if k + 1 < len(lines) else ''
                    if seg != '':
                        err(k + 1, '/solve/set/limits: o teto e o ULTIMO prompt; a '
                                   'linha seguinte devia ser comando, comentario ou '
                                   'branco, e %r parece resposta a mais' % seg)
            elif nxt == '':
                pass                                # forma de eco: so brancos
            else:
                err(i, '/solve/set/limits: esperava "," (definir) ou linha em branco '
                       '(ecoar). Recebi: %r' % nxt)

        # --- regra 4 ---
        if s.startswith('(ti-menu-load-string'):
            if i + 1 >= len(lines) or lines[i + 1].strip() != '':
                warn(i, 'ti-menu-load-string sem linha em branco depois '
                        '(rede de seguranca contra prompt pendente)')

    status = 'ERROS' if errors else ('ok, com avisos' if warnings else 'ok')
    print('%-34s %s  (%d linhas, %d erros, %d avisos)'
          % (path, status, len(lines), errors, warnings))
    return errors, warnings


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    total = 0
    for p in sys.argv[1:]:
        e, _ = check(p)
        total += e
    sys.exit(1 if total else 0)


if __name__ == '__main__':
    main()
