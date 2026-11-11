"""Active orchestration: genuine FedAvg, server-student soft labels, validation and reports."""
import argparse,fcntl,json,time,traceback
from support.federated_global.shared import ROOT,FED,SOFT,dump,signature

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['all','fedavg','soft','validate','report'],default='all')
    p.add_argument('--experiment',choices=['exp1','exp2','exp3','exp4','exp5'],action='append')
    p.add_argument('--pilot',action='store_true');args=p.parse_args()
    root=ROOT/'artifacts/global_protocol';root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Já existe uma execução do novo protocolo.')
        try:
            from support.federated_global import fedavg,soft,validation,notebooks
            def refresh_reports():
                notebooks.refresh()
                from support.experiment_plots import refresh_charts
                refresh_charts()
            if args.phase in ['all','fedavg']:
                dump(root/'status.json',dict(status='running',phase='fedavg',signature=signature()))
                fedavg.run(pilot=args.pilot)
                validation.validate_fedavg(require_complete=not args.pilot)
                refresh_reports()
            if args.phase in ['all','soft']:
                dump(root/'status.json',dict(status='running',phase='soft',signature=signature()))
                soft.run(experiments=args.experiment,pilot=args.pilot)
                validation.validate_soft(require_complete=not (args.pilot or args.experiment))
                refresh_reports()
            if args.phase=='validate':
                validation.validate_fedavg(require_complete=not args.pilot)
                validation.validate_soft(require_complete=not args.pilot)
            if args.phase=='report':refresh_reports()
            dump(root/'status.json',dict(status='pilot_complete' if args.pilot else 'requested_phase_complete',phase=args.phase,signature=signature()))
        except Exception as e:
            dump(root/'status.json',dict(status='failed',phase=args.phase,error=str(e)))
            traceback.print_exc();raise
if __name__=='__main__':main()
