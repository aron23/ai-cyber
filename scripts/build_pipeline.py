#!/usr/bin/env python3

import yaml
import logging
import subprocess
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BuildPipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.build_id = f"build_{int(time.time())}"
        self.artifacts = {}
        self.stage_results = {}
        
    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _substitute_variables(self, value: str) -> str:
        if isinstance(value, str) and '${' in value:
            # Simple variable substitution
            for var in ['DATA_PATH', 'VERSION', 'BUILD_ID', 'DOCKER_REGISTRY', 
                       'TEAM_EMAIL', 'SLACK_CHANNEL']:
                env_value = os.environ.get(var, '')
                value = value.replace(f'${{{var}}}', env_value)
            value = value.replace('${BUILD_ID}', self.build_id)
        return value
    
    def _run_step(self, stage_name: str, step: Dict[str, Any]) -> Dict[str, Any]:
        step_name = step['name']
        logger.info(f"Running step: {stage_name}.{step_name}")
        
        start_time = time.time()
        
        try:
            # Prepare script command
            script = step['script']
            cmd = ['python', script]
            
            # Add inputs as arguments
            if 'inputs' in step:
                for key, value in step['inputs'].items():
                    value = self._substitute_variables(value)
                    cmd.extend([f'--{key}', value])
            
            # Add outputs as arguments
            if 'outputs' in step:
                for key, value in step['outputs'].items():
                    value = self._substitute_variables(value)
                    cmd.extend([f'--output-{key}', value])
                    # Store artifact paths
                    self.artifacts[f"{stage_name}.{step_name}.{key}"] = value
            
            # Run the command
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Step failed: {result.stderr}")
            
            duration = time.time() - start_time
            
            return {
                'step': step_name,
                'status': 'success',
                'duration': duration,
                'outputs': step.get('outputs', {})
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Step {step_name} failed: {str(e)}")
            return {
                'step': step_name,
                'status': 'failed',
                'duration': duration,
                'error': str(e)
            }
    
    def _run_stage(self, stage: Dict[str, Any]) -> Dict[str, Any]:
        stage_name = stage['name']
        logger.info(f"Starting stage: {stage_name}")
        
        start_time = time.time()
        results = []
        
        if stage.get('parallel', False):
            # Run steps in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._run_step, stage_name, step): step
                    for step in stage['steps']
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if result['status'] == 'failed':
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
        else:
            # Run steps sequentially
            for step in stage['steps']:
                result = self._run_step(stage_name, step)
                results.append(result)
                if result['status'] == 'failed':
                    break
        
        duration = time.time() - start_time
        stage_status = 'success' if all(r['status'] == 'success' for r in results) else 'failed'
        
        return {
            'stage': stage_name,
            'status': stage_status,
            'duration': duration,
            'steps': results
        }
    
    def run(self) -> Dict[str, Any]:
        logger.info(f"Starting build pipeline: {self.build_id}")
        
        start_time = time.time()
        pipeline_results = {
            'build_id': self.build_id,
            'config': self.config_path,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'stages': []
        }
        
        try:
            # Run each stage
            for stage in self.config['stages']:
                stage_result = self._run_stage(stage)
                pipeline_results['stages'].append(stage_result)
                self.stage_results[stage['name']] = stage_result
                
                if stage_result['status'] == 'failed':
                    logger.error(f"Stage {stage['name']} failed. Stopping pipeline.")
                    break
            
            # Calculate overall status
            all_success = all(
                s['status'] == 'success' for s in pipeline_results['stages']
            )
            pipeline_results['status'] = 'success' if all_success else 'failed'
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            pipeline_results['status'] = 'failed'
            pipeline_results['error'] = str(e)
        
        # Add completion info
        duration = time.time() - start_time
        pipeline_results['duration'] = duration
        pipeline_results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        pipeline_results['artifacts'] = self.artifacts
        
        # Save results
        self._save_results(pipeline_results)
        
        # Send notifications
        self._send_notifications(pipeline_results)
        
        return pipeline_results
    
    def _save_results(self, results: Dict[str, Any]):
        results_dir = Path('build_results')
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"{self.build_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save as latest
        latest_file = results_dir / 'latest_build.json'
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Build results saved to {results_file}")
    
    def _send_notifications(self, results: Dict[str, Any]):
        status = results['status']
        notifications = self.config.get('notifications', {})
        
        if status == 'success' and 'on_success' in notifications:
            for notification in notifications['on_success']:
                self._send_notification(notification, results)
        elif status == 'failed' and 'on_failure' in notifications:
            for notification in notifications['on_failure']:
                self._send_notification(notification, results)
    
    def _send_notification(self, notification: Dict[str, Any], results: Dict[str, Any]):
        notif_type = notification['type']
        
        if notif_type == 'email':
            # In a real implementation, send email
            logger.info(f"Would send email to {notification['to']}: {notification['subject']}")
        elif notif_type == 'slack':
            # In a real implementation, send Slack message
            logger.info(f"Would send Slack message to {notification['channel']}: {notification['message']}")


def create_stub_scripts():
    scripts_dir = Path('scripts')
    scripts_dir.mkdir(exist_ok=True)
    
    stub_scripts = [
        'validate_data.py', 'preprocess_data.py', 'feature_engineering.py',
        'train_cnn.py', 'train_transformer.py', 'train_ensemble.py',
        'run_tests.py', 'regression_tests.py', 'ab_tests.py',
        'select_best_model.py', 'optimize_model.py', 'package_model.py'
    ]
    
    for script in stub_scripts:
        script_path = scripts_dir / script
        if not script_path.exists():
            with open(script_path, 'w') as f:
                f.write(f'''#!/usr/bin/env python3
"""Stub script for {script}"""
import argparse
import time
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    # Add generic arguments
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--validated_data', type=str)
    parser.add_argument('--processed_data', type=str)
    parser.add_argument('--feature_data', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--models_dir', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--baseline', type=str)
    parser.add_argument('--test_config', type=str)
    parser.add_argument('--test_results', type=str)
    parser.add_argument('--regression_results', type=str)
    parser.add_argument('--ab_results', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--requirements', type=str)
    
    # Add output arguments
    for output in ['validated_data', 'processed_data', 'feature_data', 
                   'model', 'metrics', 'test_results', 'regression_results',
                   'ab_results', 'best_model', 'selection_report', 
                   'optimized_model', 'package', 'docker_image']:
        parser.add_argument(f'--output-{output}', type=str)
    
    args = parser.parse_args()
    
    print(f"Running {script}...")
    time.sleep(1)  # Simulate work
    
    # Create dummy outputs
    for arg, value in vars(args).items():
        if arg.startswith('output-') and value:
            output_path = Path(value)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if value.endswith('.json'):
                with open(output_path, 'w') as f:
                    json.dump({{"status": "success", "script": "{script}"}}, f)
            else:
                output_path.touch()
    
    print(f"{script} completed successfully")

if __name__ == "__main__":
    main()
''')
            script_path.chmod(0o755)


def main():
    parser = argparse.ArgumentParser(description='Run build pipeline')
    parser.add_argument('--config', type=str, default='build/build_models.yml',
                       help='Path to build configuration file')
    parser.add_argument('--create-stubs', action='store_true',
                       help='Create stub scripts for testing')
    
    args = parser.parse_args()
    
    if args.create_stubs:
        create_stub_scripts()
        logger.info("Stub scripts created")
        return
    
    # Set some example environment variables
    os.environ['DATA_PATH'] = 'data/raw/'
    os.environ['VERSION'] = '1.0.0'
    os.environ['DOCKER_REGISTRY'] = 'gcr.io/project-id'
    os.environ['TEAM_EMAIL'] = 'team@example.com'
    os.environ['SLACK_CHANNEL'] = '#ml-builds'
    
    # Run pipeline
    pipeline = BuildPipeline(args.config)
    results = pipeline.run()
    
    # Print summary
    print(f"\nBuild {results['build_id']} completed with status: {results['status']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    
    if results['status'] == 'success':
        print("\nArtifacts produced:")
        for name, path in results['artifacts'].items():
            print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()