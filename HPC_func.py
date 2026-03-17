import submitit
from utils import *
import shutil
from concurrent.futures import ThreadPoolExecutor

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)

T = Tools_Extend()


def sumbit_jobs_array_with_pbar(params_list):
    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)
    if os.path.exists(progress_dir_json):
        shutil.rmtree(progress_dir_json)
    T.mkdir(progress_dir_json, force=True)

    print('submiting...')
    pass


def sumbit_jobs_array(func,params_list,log_folder,job_name,
                        job_number_limit=500,
                        parallel_process_per_task=10,
                        slurm_array_parallelism=20,
                        parallel_process_p_or_t='p',
                        cpus_per_task=1,
                        mem_gb=1,
                        timeout_min=5,
                        slurm_partition="general",
                      ):
    if len(params_list) == 0:
        raise ValueError("params_list is empty")
    if len(params_list) > job_number_limit:
        super_params_list = T.split_into_n_jobs(params_list,job_number_limit)

        def super_func(chunk):
            def wrapper(p):
                return func(p)

            if parallel_process_p_or_t == 'p':
                with multiprocessing.Pool(parallel_process_per_task) as P:
                    P.map(wrapper, chunk)
            elif parallel_process_p_or_t == 't':
                with ThreadPoolExecutor(max_workers=parallel_process_per_task) as Thread_:
                    list(Thread_.map(wrapper, chunk))

        final_params_list = super_params_list
        final_func = super_func
    else:
        final_params_list = params_list
        final_func = func

    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)
    T.mkdir(log_folder, force=True)

    # print('submiting...')
    # exit()
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name=job_name,
        cpus_per_task=cpus_per_task,
        mem_gb=mem_gb,
        timeout_min=timeout_min,
        slurm_array_parallelism=slurm_array_parallelism,
        slurm_partition=slurm_partition,
    )
    jobs = executor.map_array(final_func, final_params_list)
    print('total param len:', len(params_list))
    print('len(jobs):', len(final_params_list))
    print('jobid,',jobs[0].job_id)


def monitoring_job(progress_dir_json):
    refresh_interval = 1
    # refresh_interval = 5
    task_map = {}
    with Progress(
            TextColumn("[bold yellow]{task.description}"),
            BarColumn(style="red",  # Unfinished part
                      complete_style="green",  # Finished part
                      finished_style="yellow"),  # 100% finished),
            TextColumn("{task.completed}/{task.total} {task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
    ) as progress_total:

        with Progress(
                TextColumn("[bold yellow]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total} {task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
        ) as progress:

            overall_task = None
            total_jobs = None

            while True:

                files = list(progress_dir_json.glob("*.json"))
                finished_jobs = 0

                for f in files:

                    try:
                        with open(f) as fp:
                            data = json.load(fp)
                    except:
                        continue

                    jobid = data["jobid"]
                    step = data["step"]
                    total = data["total"]
                    total_job = data["total_job"]
                    sub_job_name = data["sub_job_name"]

                    # 初始化 total progress

                    if total_jobs is None:
                        total_jobs = total_job
                        overall_task = progress_total.add_task(
                            "[bold green]TOTAL", total=total_jobs
                        )

                    # 判断是否完成
                    if step >= total:

                        finished_jobs += 1

                        if jobid in task_map:
                            progress.remove_task(task_map[jobid])
                            del task_map[jobid]

                        continue

                    # 创建 bar
                    if jobid not in task_map:
                        task_map[jobid] = progress.add_task(
                            f"({jobid}/{total_job}) {sub_job_name}", total=total
                        )

                    progress.update(task_map[jobid], completed=step)

                # 更新 total bar
                if overall_task is not None:
                    progress_total.update(overall_task, completed=finished_jobs)

                # 所有任务完成
                if total_jobs and finished_jobs == total_jobs:
                    break

                time.sleep(refresh_interval)
    pass

