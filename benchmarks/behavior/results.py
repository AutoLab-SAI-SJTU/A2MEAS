"""Summarize completed BEHAVIOR runs using their actual evaluation plan."""
import argparse
import json
import math
from pathlib import Path

from a2meas.behavior import load_routes


TIMEOUT_FAILURES = ('group_time_budget_exceeded', 'policy_query_timeout')


def validate_result(result, task, instance):
    if (not isinstance(result, dict) or result.get('task') != task or result.get('instance_id') != instance
            or result.get('rollout_id') != 0 or type(result.get('success')) is not bool
            or result.get('failure_reason') not in (None, '', *TIMEOUT_FAILURES)):
        raise ValueError(f'Invalid evaluation result: {task}/{instance}')
    q_score = result.get('q_score')
    score = q_score.get('final') if isinstance(q_score, dict) else None
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f'Invalid Q-score: {task}/{instance}')
    if result.get('failure_reason') in TIMEOUT_FAILURES and (result['success'] or score != 0):
        raise ValueError(f'Timeout must be a failed zero-score result: {task}/{instance}')
    return result


def read_result(path, task, instance):
    return validate_result(json.loads(path.read_text()), task, instance)


def summarize(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'run.json').read_text())
    plan = manifest['phases']
    mode = manifest['identity']['mode']
    roots = list((run_dir / 'evaluations').glob('*/json'))
    if len(roots) != 1:
        raise ValueError('Expected one evaluation JSON directory')
    result_dir = roots[0]
    routes = load_routes()['tasks']
    tasks, missing = [], []
    for phase in plan:
        for job in phase['tasks']:
            task = routes[str(job['id'])]['task']
            successes, scores = 0, []
            for index in job['instances']:
                instance = (301 if mode == 'public_test' else 321) + index
                path = result_dir / f'{task}_{instance}_0.json'
                if not path.is_file():
                    missing.append(str(path))
                    continue
                result = read_result(path, task, instance)
                score = result['q_score']['final']
                successes += result['success']
                scores.append(score)
            if len(scores) == len(job['instances']):
                tasks.append(dict(id=job['id'], task=task, episodes=len(scores), successes=successes,
                                  success_rate=successes / len(scores), mean_q_score=sum(scores) / len(scores)))
    if missing:
        raise ValueError(f'Missing {len(missing)} planned result(s); first: {missing[0]}')
    total = sum(task['episodes'] for task in tasks)
    return dict(episodes=total, successes=sum(task['successes'] for task in tasks),
                success_rate=sum(task['successes'] for task in tasks) / total,
                mean_task_success_rate=sum(task['success_rate'] for task in tasks) / len(tasks),
                mean_task_q_score=sum(task['mean_q_score'] for task in tasks) / len(tasks), tasks=tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir), indent=2))


if __name__ == '__main__':
    main()
